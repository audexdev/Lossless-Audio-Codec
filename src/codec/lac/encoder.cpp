#include "encoder.hpp"
#include <limits>
#include <stdexcept>
#include <string>
#include <atomic>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include "codec/simd/neon.hpp"
#include "codec/lpc/lpc.hpp"
#include "utils/logger.hpp"

namespace {
  constexpr double kAbsDiffWeight = 1.0;
  constexpr double kVarianceWeight = 0.5;
  constexpr double kSlopePenaltyWeight = 0.25;
  constexpr double kSlopeThreshold = 4.0;
  constexpr uint64_t kMaxCostProxy = std::numeric_limits<uint64_t>::max() / 4;

  struct BlockInfo {
    size_t start;
    uint32_t size;
  };

  bool has_zero_run_samples(const std::vector<int32_t>& samples) {
    if (samples.empty()) return false;
    size_t idx = 0;
    while (idx < samples.size()) {
      if (samples[idx] == 0) {
        size_t run = 0;
        while (idx + run < samples.size() && samples[idx + run] == 0) {
          ++run;
        }
        if (run >= Block::ZERO_RUN_MIN_LENGTH) return true;
        idx += run;
      } else {
        ++idx;
      }
    }
    return false;
  }

  bool should_enable_zero_run_for_block(const std::vector<int32_t>& pcm, int order) {
    if (pcm.empty()) return false;
    if (!has_zero_run_samples(pcm)) return false;
    Block::Encoder estimator(order, false, false);
    estimator.set_zero_run_enabled(true);
    uint64_t bits_normal = 0;
    uint64_t bits_zr = 0;
    uint64_t bits_bin = 0;
    if (!estimator.estimate_bits(pcm, bits_normal, bits_zr, bits_bin)) {
      return false;
    }
    return bits_zr < std::min(bits_normal, bits_bin);
  }

  double block_slope_penalty(const std::vector<int32_t>& channel, size_t position, size_t length) {
    if (channel.empty() || length <= 1 || position + length > channel.size()) return 0.0;
    const int64_t first = channel[position];
    const int64_t last = channel[position + length - 1];
    const int64_t delta = (last >= first) ? (last - first) : (first - last);
    const double slope = static_cast<double>(delta) / static_cast<double>(length);
    if (slope <= kSlopeThreshold) {
      return 0.0;
    }
    const double size_scale = static_cast<double>(length) / static_cast<double>(Block::MAX_BLOCK_SIZE);
    return (slope - kSlopeThreshold) * kSlopePenaltyWeight * size_scale;
  }

  inline uint64_t estimate_rice_bits(const std::vector<int32_t>& residual) {
    if (residual.empty()) return 0;
    uint64_t sum_u = 0;
    for (int32_t r : residual) {
      sum_u += (static_cast<uint32_t>(r) << 1) ^ (static_cast<uint32_t>(r >> 31));
    }
    const uint64_t mean = (sum_u + (residual.size() >> 1)) / residual.size();
    uint32_t k = 0;
    while ((1u << k) < mean && k < 31u) ++k;

    uint64_t bits = 0;
    for (int32_t r : residual) {
      const uint32_t u = (static_cast<uint32_t>(r) << 1) ^ (static_cast<uint32_t>(r >> 31));
      const uint32_t q = (k >= 31u) ? 0u : (u >> k);
      bits += static_cast<uint64_t>(q) + 1u + k;
    }
    return bits;
  }

  struct StereoCost {
    uint64_t lr_bits = kMaxCostProxy;
    uint64_t ms_bits = kMaxCostProxy;
    bool lr_valid = false;
    bool ms_valid = false;
    bool choose_ms() const {
      if (ms_valid && !lr_valid) return true;
      if (lr_valid && !ms_valid) return false;
      if (!lr_valid && !ms_valid) return false;
      if (ms_bits < lr_bits) return true;
      if (ms_bits > lr_bits) return false;
      return false; // tie -> LR
    }
  };

  bool estimate_channel_cost(const std::vector<int32_t>& pcm,
      int order,
      bool zero_run_enabled,
      bool partitioning_enabled,
      uint64_t& bits) {
    Block::Encoder estimator(order, false, false);
    const bool use_zr = zero_run_enabled && should_enable_zero_run_for_block(pcm, order);
    estimator.set_zero_run_enabled(use_zr);
    estimator.set_partitioning_enabled(partitioning_enabled);
    uint64_t bits_normal = 0;
    uint64_t bits_zr = 0;
    uint64_t bits_bin = 0;
    if (!estimator.estimate_bits(pcm, bits_normal, bits_zr, bits_bin)) {
      return false;
    }
    bits = std::min(bits_normal, std::min(bits_zr, bits_bin));
    return true;
  }

  StereoCost estimate_stereo_cost(const std::vector<int32_t>& left,
      const std::vector<int32_t>& right,
      size_t start,
      size_t size,
      int order,
      bool zero_run_enabled,
      bool partitioning_enabled) {
    StereoCost cost;
    if (size == 0 || start + size > left.size() || start + size > right.size()) {
      return cost;
    }

    std::vector<int32_t> blockL(left.begin() + start, left.begin() + start + size);
    std::vector<int32_t> blockR(right.begin() + start, right.begin() + start + size);
    cost.lr_valid = estimate_channel_cost(blockL,
                                          order,
                                          zero_run_enabled,
                                          partitioning_enabled,
                                          cost.lr_bits);
    uint64_t bits_r = 0;
    if (estimate_channel_cost(blockR,
                              order,
                              zero_run_enabled,
                              partitioning_enabled,
                              bits_r)) {
      if (cost.lr_valid) {
        cost.lr_bits += bits_r;
      } else {
        cost.lr_bits = bits_r;
        cost.lr_valid = true;
      }
    } else {
      cost.lr_valid = false;
    }

    // MS path
    std::vector<int32_t> blockM(size);
    std::vector<int32_t> blockS(size);
    SIMD::ms_encode_simd_or_scalar(
        left.data() + start,
        right.data() + start,
        blockM.data(),
        blockS.data(),
        size
        );
    cost.ms_valid = estimate_channel_cost(blockM,
                                          order,
                                          zero_run_enabled,
                                          partitioning_enabled,
                                          cost.ms_bits);
    uint64_t bits_s = 0;
    if (estimate_channel_cost(blockS,
                              order,
                              zero_run_enabled,
                              partitioning_enabled,
                              bits_s)) {
      if (cost.ms_valid) {
        cost.ms_bits += bits_s;
      } else {
        cost.ms_bits = bits_s;
        cost.ms_valid = true;
      }
    } else {
      cost.ms_valid = false;
    }

    return cost;
  }
}

namespace LAC {

  Encoder::Encoder(uint8_t order, uint8_t stereo_mode, uint32_t sample_rate, uint8_t bit_depth, bool debug_lpc, bool debug_stereo_est, bool debug_zr)
    : order(order),
    stereo_mode(stereo_mode),
    sample_rate(sample_rate),
    bit_depth(bit_depth),
    debug_lpc(debug_lpc),
    debug_stereo_est(debug_stereo_est),
    debug_zr(debug_zr),
    zero_run_enabled(true),
    partitioning_enabled(true),
    debug_partitions(false),
    candidates{256, 512, 1024, 2048, 4096, 8192, 16384} {}

  std::vector<uint8_t> Encoder::encode(
      const std::vector<int32_t>& left,
      const std::vector<int32_t>& right,
      ThreadCollector* collector
      ) {
    if (!right.empty() && right.size() != left.size()) {
      throw std::invalid_argument(
          "right channel size (" + std::to_string(right.size()) +
          ") must match left channel size (" + std::to_string(left.size()) + ")"
          );
    }

    BitWriter writer;

    FrameHeader hdr;
    hdr.channels = (right.empty() ? 1 : 2);
    hdr.stereo_mode = (hdr.channels == 2 ? this->stereo_mode : 0);
    hdr.sample_rate = this->sample_rate;
    hdr.bit_depth = this->bit_depth;
    hdr.write(writer);

    std::vector<BlockInfo> blocks;
    const size_t totalSamples = left.size();
    size_t pos = 0;
    while (pos < totalSamples) {
      uint32_t block_size = this->select_block_size(left, right, pos);
      if (block_size == 0) {
        block_size = static_cast<uint32_t>(totalSamples - pos);
      }
      blocks.push_back({pos, block_size});
      pos += block_size;
    }

    writer.write_bits(static_cast<uint32_t>(blocks.size()), 32);
    for (const auto& block : blocks) {
      writer.write_bits(block.size, 32);
    }

    const bool isStereo = (hdr.channels == 2);
    const bool forceMidSide = isStereo && (hdr.stereo_mode == 1);
    const bool perBlockStereo = isStereo && (hdr.stereo_mode == 2);
    const uint8_t globalStereoMode = hdr.stereo_mode;

    std::vector<std::vector<uint8_t>> encodedBlocks(blocks.size());
    std::queue<size_t> task_queue;
    for (size_t i = 0; i < blocks.size(); ++i) {
      task_queue.push(i);
    }

    std::mutex queue_mutex;
    std::mutex error_mutex;
    std::exception_ptr worker_error;
    std::atomic<bool> stop_requested{false};

    auto encode_block = [this, &left, &right, &blocks, isStereo, forceMidSide, perBlockStereo, globalStereoMode](size_t block_idx) -> std::vector<uint8_t> {
      const auto& block = blocks[block_idx];
      Block::Encoder blockEnc(this->order, this->debug_lpc, this->debug_zr);
      blockEnc.set_zero_run_enabled(this->zero_run_enabled);
      blockEnc.set_partitioning_enabled(this->partitioning_enabled);
      blockEnc.set_debug_partitions(this->debug_partitions);
      blockEnc.set_debug_block_index(block_idx);
      std::vector<uint8_t> encodedBytes;
      const size_t start = block.start;
      const size_t end = start + block.size;

      if (block.size == 0) {
        return encodedBytes;
      }

      auto encode_lr = [&]() -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        std::vector<int32_t> blockL(left.begin() + start, left.begin() + end);
        bool use_zr_l = this->zero_run_enabled && should_enable_zero_run_for_block(blockL, this->order);
        blockEnc.set_zero_run_enabled(use_zr_l);
        std::vector<uint8_t> encodedL = blockEnc.encode(blockL);
        out.insert(out.end(), encodedL.begin(), encodedL.end());
        if (isStereo) {
          std::vector<int32_t> blockR(right.begin() + start, right.begin() + end);
          bool use_zr_r = this->zero_run_enabled && should_enable_zero_run_for_block(blockR, this->order);
          blockEnc.set_zero_run_enabled(use_zr_r);
          std::vector<uint8_t> encodedR = blockEnc.encode(blockR);
          out.insert(out.end(), encodedR.begin(), encodedR.end());
        }
        return out;
      };

      auto encode_ms = [&]() -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        std::vector<int32_t> blockMid(block.size);
        std::vector<int32_t> blockSide(block.size);
        SIMD::ms_encode_simd_or_scalar(
            left.data() + start,
            right.data() + start,
            blockMid.data(),
            blockSide.data(),
            block.size
            );
        bool use_zr_mid = this->zero_run_enabled && should_enable_zero_run_for_block(blockMid, this->order);
        blockEnc.set_zero_run_enabled(use_zr_mid);
        std::vector<uint8_t> encodedMid = blockEnc.encode(blockMid);
        bool use_zr_side = this->zero_run_enabled && should_enable_zero_run_for_block(blockSide, this->order);
        blockEnc.set_zero_run_enabled(use_zr_side);
        std::vector<uint8_t> encodedSide = blockEnc.encode(blockSide);
        out.insert(out.end(), encodedMid.begin(), encodedMid.end());
        out.insert(out.end(), encodedSide.begin(), encodedSide.end());
        return out;
      };

      std::string mode_used = "LR";

      if (!isStereo) {
        encodedBytes = encode_lr();
      } else if (forceMidSide) {
        mode_used = "MS";
        std::vector<uint8_t> msBytes = encode_ms();
        encodedBytes.insert(encodedBytes.end(), msBytes.begin(), msBytes.end());
      } else if (!perBlockStereo) {
        mode_used = "LR";
        std::vector<uint8_t> lrBytes = encode_lr();
        encodedBytes.insert(encodedBytes.end(), lrBytes.begin(), lrBytes.end());
      } else {
        StereoCost cost = estimate_stereo_cost(left,
                                               right,
                                               start,
                                               block.size,
                                               this->order,
                                               this->zero_run_enabled,
                                               this->partitioning_enabled);
        bool choose_ms = cost.choose_ms();
        if (this->debug_stereo_est) {
          LAC_DEBUG_LOG("[stereo-est] block=" << block_idx
            << " lr_bits=" << (cost.lr_valid ? cost.lr_bits : kMaxCostProxy)
            << " ms_bits=" << (cost.ms_valid ? cost.ms_bits : kMaxCostProxy)
            << " chosen=" << (choose_ms ? "MS" : "LR") << "\n";
          );
        }
        mode_used = (choose_ms ? "MS" : "LR");
        encodedBytes.push_back(choose_ms ? 1 : 0);
        if (choose_ms) {
          std::vector<uint8_t> msBytes = encode_ms();
          encodedBytes.insert(encodedBytes.end(), msBytes.begin(), msBytes.end());
        } else {
          std::vector<uint8_t> lrBytes = encode_lr();
          encodedBytes.insert(encodedBytes.end(), lrBytes.begin(), lrBytes.end());
        }
      }

      if (this->debug_stereo_est && isStereo) {
        LAC_DEBUG_LOG("[stereo-mode] global=" << int(globalStereoMode)
          << " block=" << block_idx
          << " mode_used=" << mode_used << "\n";
        );
      }

      return encodedBytes;
    };

    const size_t hardware_threads = std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t worker_count = std::min(hardware_threads, blocks.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back([&]() {
        if (collector) {
          collector->record(std::this_thread::get_id());
        }
        while (true) {
          if (stop_requested.load(std::memory_order_acquire)) {
            return;
          }

          size_t block_idx = 0;
          {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (task_queue.empty()) {
              return;
            }
            block_idx = task_queue.front();
            task_queue.pop();
          }

          try {
            encodedBlocks[block_idx] = encode_block(block_idx);
          } catch (...) {
            stop_requested.store(true, std::memory_order_release);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!worker_error) {
              worker_error = std::current_exception();
            }
            return;
          }
        }
      });
    }

    for (auto& worker : workers) {
      worker.join();
    }

    if (worker_error) {
      std::rethrow_exception(worker_error);
    }

    for (const auto& bytes : encodedBlocks) {
      for (uint8_t b : bytes) {
        writer.write_bits(b, 8);
      }
    }

    writer.flush_to_byte();
    return writer.get_buffer();
  }

  void Encoder::set_zero_run_enabled(bool enabled) {
    this->zero_run_enabled = enabled;
  }

  void Encoder::set_partitioning_enabled(bool enabled) {
    this->partitioning_enabled = enabled;
  }

  void Encoder::set_debug_partitions(bool enabled) {
    this->debug_partitions = enabled;
  }

  uint32_t Encoder::select_block_size(const std::vector<int32_t>& left,
      const std::vector<int32_t>& right,
      size_t position) const {
    if (position >= left.size()) return 0;
    size_t remaining = left.size() - position;
    if (remaining == 0) return 0;

    double best_cost = std::numeric_limits<double>::infinity();
    uint32_t best_size = static_cast<uint32_t>(std::min<size_t>(remaining, Block::MAX_BLOCK_SIZE));
    bool evaluated = false;

    for (uint32_t candidate : this->candidates) {
      if (candidate == 0 || candidate > remaining) continue;
      double cost = this->block_complexity(left, position, candidate);
      if (!right.empty()) {
        cost += this->block_complexity(right, position, candidate);
      }
      cost += block_slope_penalty(left, position, candidate);
      if (!right.empty()) {
        cost += block_slope_penalty(right, position, candidate);
      }
      evaluated = true;
      if (cost < best_cost || (cost == best_cost && candidate > best_size)) {
        best_cost = cost;
        best_size = candidate;
      }
    }

    if (!evaluated) {
      best_size = static_cast<uint32_t>(std::min<size_t>(remaining, Block::MAX_BLOCK_SIZE));
    }

    if (best_size > remaining) {
      best_size = static_cast<uint32_t>(remaining);
    }
    if (best_size > Block::MAX_BLOCK_SIZE) {
      best_size = Block::MAX_BLOCK_SIZE;
    }
    if (best_size == 0) {
      best_size = static_cast<uint32_t>(std::min<size_t>(remaining, Block::MAX_BLOCK_SIZE));
    }

    return best_size;
  }

  double Encoder::block_complexity(const std::vector<int32_t>& channel,
      size_t position,
      size_t length) const {
    if (channel.empty() || length <= 1) return 0.0;
    if (position + length > channel.size()) length = channel.size() - position;
    if (length <= 1) return 0.0;

    int64_t absdiff_sum = 0;
    int64_t sum = 0;
    long double sumsq = 0.0L;

    const size_t end = position + length;
    const int32_t first = channel[position];
    sum = static_cast<int64_t>(first);
    sumsq = static_cast<long double>(first) * static_cast<long double>(first);

    if constexpr (SIMD::kHasNeon) {
      absdiff_sum = SIMD::neon_absdiff_sum(channel.data() + position, length);
      for (size_t i = position + 1; i < end; ++i) {
        const int64_t v = static_cast<int64_t>(channel[i]);
        sum += v;
        sumsq += static_cast<long double>(v) * static_cast<long double>(v);
      }
    } else {
      int32_t prev = first;
      for (size_t i = position + 1; i < end; ++i) {
        const int32_t cur = channel[i];
        const int64_t v = static_cast<int64_t>(cur);
        sum += v;
        sumsq += static_cast<long double>(v) * static_cast<long double>(v);
        const int64_t diff = static_cast<int64_t>(cur) - static_cast<int64_t>(prev);
        absdiff_sum += (diff >= 0 ? diff : -diff);
        prev = cur;
      }
    }

    const double absdiff_avg = static_cast<double>(absdiff_sum) / static_cast<double>(length);
    const double mean = static_cast<double>(sum) / static_cast<double>(length);
    const double mean_sq = mean * mean;
    const double avg_sq = static_cast<double>(sumsq / static_cast<long double>(length));
    double variance = avg_sq - mean_sq;
    if (variance < 0.0) variance = 0.0;

    return (kAbsDiffWeight * absdiff_avg) + (kVarianceWeight * variance);
  }

} // namespace LAC
