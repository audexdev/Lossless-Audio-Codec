#include "encoder.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <iostream>
#include "codec/simd/neon.hpp"
#include "codec/lac/thread_limit.hpp"
#include "utils/logger.hpp"

namespace {
  constexpr uint64_t kStereoConfidenceDivisor = 100;
  constexpr size_t kStereoProbeSize = 256;
  constexpr size_t kStereoFullComparisonLimit = 4096;
  constexpr int32_t kPcm16Min = -32768;
  constexpr int32_t kPcm16Max = 32767;
  constexpr int32_t kPcm24Min = -0x800000;
  constexpr int32_t kPcm24Max = 0x7FFFFF;

  struct BlockInfo {
    size_t start;
    uint32_t size;
  };

  uint64_t add_saturated(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
      return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
  }

  uint64_t zigzag_difference(int64_t value) {
    if (value >= 0) return static_cast<uint64_t>(value) << 1;
    return (static_cast<uint64_t>(-(value + 1)) << 1) | 1u;
  }

  uint32_t rice_k_for_mean(uint64_t sum, uint64_t count) {
    if (count == 0) return 0;
    const uint64_t mean = (sum + (count >> 1)) / count;
    uint32_t k = 0;
    while (k < 31u && (uint64_t{1} << k) < mean) {
      ++k;
    }
    return k;
  }

  uint64_t approximate_rice_bits(uint64_t sum, uint64_t count) {
    if (count == 0) return 0;
    const uint32_t k = rice_k_for_mean(sum, count);
    return add_saturated(sum >> k, count * static_cast<uint64_t>(k + 1u));
  }

  std::vector<BlockInfo> plan_blocks(const std::vector<int32_t>& left) {
    std::vector<BlockInfo> blocks;
    size_t pos = 0;
    while (pos < left.size()) {
      const uint32_t size = static_cast<uint32_t>(
          std::min<size_t>(Block::MAX_BLOCK_SIZE, left.size() - pos));
      blocks.push_back({pos, size});
      pos += size;
    }
    return blocks;
  }

  bool is_supported_sample_rate(uint32_t sample_rate) {
    return sample_rate == 44100 ||
           sample_rate == 48000 ||
           sample_rate == 96000 ||
           sample_rate == 192000;
  }

  bool is_supported_bit_depth(uint8_t bit_depth) {
    return bit_depth == 16 || bit_depth == 24;
  }

  bool is_valid_sample_for_depth(int32_t sample, uint8_t bit_depth) {
    if (bit_depth == 16) {
      return sample >= kPcm16Min && sample <= kPcm16Max;
    }
    if (bit_depth == 24) {
      return sample >= kPcm24Min && sample <= kPcm24Max;
    }
    return false;
  }

  void validate_samples_for_depth(const std::vector<int32_t>& samples,
                                  uint8_t bit_depth,
                                  const char* channel) {
    for (size_t i = 0; i < samples.size(); ++i) {
      if (!is_valid_sample_for_depth(samples[i], bit_depth)) {
        throw std::invalid_argument(
            std::string(channel) + " sample at index " + std::to_string(i) +
            " is outside the configured PCM bit depth");
      }
    }
  }

  struct StereoDecision {
    bool choose_ms = false;
    bool uncertain = false;
  };

  struct ChannelProxyCost {
    uint64_t bits;
    bool non_difference_predictor_active;
  };

  ChannelProxyCost estimate_channel_proxy_cost(uint64_t raw_sum,
                                               uint64_t diff_sum,
                                               uint64_t anti_diff_sum,
                                               uint64_t count) {
    const uint64_t raw_bits = approximate_rice_bits(raw_sum, count);
    const uint64_t diff_bits = approximate_rice_bits(diff_sum, count);
    const uint64_t anti_diff_bits = approximate_rice_bits(anti_diff_sum, count);
    return {
        std::min({raw_bits, diff_bits, anti_diff_bits}),
        raw_bits < diff_bits || anti_diff_bits < diff_bits};
  }

  StereoDecision estimate_stereo_mode(const std::vector<int32_t>& left,
      const std::vector<int32_t>& right,
      size_t start,
      size_t size) {
    uint64_t l_raw_sum = 0;
    uint64_t r_raw_sum = 0;
    uint64_t m_raw_sum = 0;
    uint64_t s_raw_sum = 0;
    uint64_t l_sum = 0;
    uint64_t r_sum = 0;
    uint64_t m_sum = 0;
    uint64_t s_sum = 0;
    uint64_t l_anti_sum = 0;
    uint64_t r_anti_sum = 0;
    uint64_t m_anti_sum = 0;
    uint64_t s_anti_sum = 0;
    int64_t previous_l = 0;
    int64_t previous_r = 0;
    int64_t previous_m = 0;
    int64_t previous_s = 0;
    for (size_t i = 0; i < size; ++i) {
      const int64_t l = left[start + i];
      const int64_t r = right[start + i];
      const int64_t m = (l + r) >> 1;
      const int64_t s = l - r;
      l_raw_sum = add_saturated(l_raw_sum, zigzag_difference(l));
      r_raw_sum = add_saturated(r_raw_sum, zigzag_difference(r));
      m_raw_sum = add_saturated(m_raw_sum, zigzag_difference(m));
      s_raw_sum = add_saturated(s_raw_sum, zigzag_difference(s));
      if (i == 0) {
        l_sum = zigzag_difference(l);
        r_sum = zigzag_difference(r);
        m_sum = zigzag_difference(m);
        s_sum = zigzag_difference(s);
        l_anti_sum = l_sum;
        r_anti_sum = r_sum;
        m_anti_sum = m_sum;
        s_anti_sum = s_sum;
      } else {
        l_sum = add_saturated(l_sum, zigzag_difference(l - previous_l));
        r_sum = add_saturated(r_sum, zigzag_difference(r - previous_r));
        m_sum = add_saturated(m_sum, zigzag_difference(m - previous_m));
        s_sum = add_saturated(s_sum, zigzag_difference(s - previous_s));
        l_anti_sum = add_saturated(l_anti_sum, zigzag_difference(l + previous_l));
        r_anti_sum = add_saturated(r_anti_sum, zigzag_difference(r + previous_r));
        m_anti_sum = add_saturated(m_anti_sum, zigzag_difference(m + previous_m));
        s_anti_sum = add_saturated(s_anti_sum, zigzag_difference(s + previous_s));
      }
      previous_l = l;
      previous_r = r;
      previous_m = m;
      previous_s = s;
    }
    const uint64_t count = static_cast<uint64_t>(size);
    const ChannelProxyCost l_cost = estimate_channel_proxy_cost(l_raw_sum, l_sum, l_anti_sum, count);
    const ChannelProxyCost r_cost = estimate_channel_proxy_cost(r_raw_sum, r_sum, r_anti_sum, count);
    const ChannelProxyCost m_cost = estimate_channel_proxy_cost(m_raw_sum, m_sum, m_anti_sum, count);
    const ChannelProxyCost s_cost = estimate_channel_proxy_cost(s_raw_sum, s_sum, s_anti_sum, count);
    const uint64_t lr_bits = add_saturated(l_cost.bits, r_cost.bits);
    const uint64_t ms_bits = add_saturated(m_cost.bits, s_cost.bits);
    const uint64_t smaller = std::min(lr_bits, ms_bits);
    const uint64_t difference = (lr_bits >= ms_bits) ? (lr_bits - ms_bits) : (ms_bits - lr_bits);
    const bool non_difference_predictor_active =
        l_cost.non_difference_predictor_active || r_cost.non_difference_predictor_active ||
        m_cost.non_difference_predictor_active || s_cost.non_difference_predictor_active;
    StereoDecision decision;
    decision.choose_ms = ms_bits < lr_bits;
    decision.uncertain =
        smaller == 0 || difference == 0 || non_difference_predictor_active ||
        difference <= smaller / kStereoConfidenceDivisor;
    return decision;
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
    thread_count(0) {}

  std::vector<uint8_t> Encoder::encode(
      const std::vector<int32_t>& left,
      const std::vector<int32_t>& right,
      ThreadCollector* collector
      ) {
    if (left.empty()) {
      throw std::invalid_argument("left channel must not be empty");
    }
    if (!right.empty() && right.size() != left.size()) {
      throw std::invalid_argument(
          "right channel size (" + std::to_string(right.size()) +
          ") must match left channel size (" + std::to_string(left.size()) + ")"
          );
    }
    if (!is_supported_sample_rate(this->sample_rate)) {
      throw std::invalid_argument("unsupported sample rate: " + std::to_string(this->sample_rate));
    }
    if (!is_supported_bit_depth(this->bit_depth)) {
      throw std::invalid_argument("unsupported bit depth: " + std::to_string(this->bit_depth));
    }
    if (this->stereo_mode > 2) {
      throw std::invalid_argument("unsupported stereo mode: " + std::to_string(this->stereo_mode));
    }
    validate_samples_for_depth(left, this->bit_depth, "left");
    if (!right.empty()) {
      validate_samples_for_depth(right, this->bit_depth, "right");
    }

    BitWriter writer;

    FrameHeader hdr;
    hdr.channels = (right.empty() ? 1 : 2);
    hdr.stereo_mode = (hdr.channels == 2 ? this->stereo_mode : 0);
    hdr.sample_rate = this->sample_rate;
    hdr.bit_depth = this->bit_depth;
    hdr.write(writer);

    std::vector<BlockInfo> blocks = plan_blocks(left);

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

      if (block.size == 0) {
        return encodedBytes;
      }

      auto encode_lr = [&](size_t range_start, size_t range_size) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        std::vector<int32_t> blockL(left.begin() + range_start, left.begin() + range_start + range_size);
        blockEnc.set_zero_run_enabled(this->zero_run_enabled);
        std::vector<uint8_t> encodedL = blockEnc.encode(blockL);
        out.insert(out.end(), encodedL.begin(), encodedL.end());
        if (isStereo) {
          std::vector<int32_t> blockR(right.begin() + range_start, right.begin() + range_start + range_size);
          blockEnc.set_zero_run_enabled(this->zero_run_enabled);
          std::vector<uint8_t> encodedR = blockEnc.encode(blockR);
          out.insert(out.end(), encodedR.begin(), encodedR.end());
        }
        return out;
      };

      auto encode_ms = [&](size_t range_start, size_t range_size) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        std::vector<int32_t> blockMid(range_size);
        std::vector<int32_t> blockSide(range_size);
        SIMD::ms_encode_simd_or_scalar(
            left.data() + range_start,
            right.data() + range_start,
            blockMid.data(),
            blockSide.data(),
            range_size
            );
        blockEnc.set_zero_run_enabled(this->zero_run_enabled);
        std::vector<uint8_t> encodedMid = blockEnc.encode(blockMid);
        blockEnc.set_zero_run_enabled(this->zero_run_enabled);
        std::vector<uint8_t> encodedSide = blockEnc.encode(blockSide);
        out.insert(out.end(), encodedMid.begin(), encodedMid.end());
        out.insert(out.end(), encodedSide.begin(), encodedSide.end());
        return out;
      };

      std::string mode_used = "LR";

      if (!isStereo) {
        encodedBytes = encode_lr(start, block.size);
      } else if (forceMidSide) {
        mode_used = "MS";
        std::vector<uint8_t> msBytes = encode_ms(start, block.size);
        encodedBytes.insert(encodedBytes.end(), msBytes.begin(), msBytes.end());
      } else if (!perBlockStereo) {
        mode_used = "LR";
        std::vector<uint8_t> lrBytes = encode_lr(start, block.size);
        encodedBytes.insert(encodedBytes.end(), lrBytes.begin(), lrBytes.end());
      } else {
        const StereoDecision decision = estimate_stereo_mode(left, right, start, block.size);
        bool choose_ms = decision.choose_ms;
        std::vector<uint8_t> selected;
        if (decision.uncertain) {
          if (block.size <= kStereoFullComparisonLimit) {
            std::vector<uint8_t> lrBytes = encode_lr(start, block.size);
            std::vector<uint8_t> msBytes = encode_ms(start, block.size);
            choose_ms = msBytes.size() < lrBytes.size();
            selected = choose_ms ? std::move(msBytes) : std::move(lrBytes);
          } else {
            // Spread bounded probes across the block so a local transition does not dominate mode selection.
            const size_t probe_starts[] = {
                start,
                start + (block.size - kStereoProbeSize) / 2u,
                start + block.size - kStereoProbeSize};
            size_t lr_probe_size = 0;
            size_t ms_probe_size = 0;
            for (const size_t probe_start : probe_starts) {
              lr_probe_size += encode_lr(probe_start, kStereoProbeSize).size();
              ms_probe_size += encode_ms(probe_start, kStereoProbeSize).size();
            }
            choose_ms = ms_probe_size < lr_probe_size;
          }
        }
        if (this->debug_stereo_est) {
          LAC_DEBUG_LOG("[stereo-est] block=" << block_idx
            << " uncertain=" << (decision.uncertain ? 1 : 0)
            << " chosen=" << (choose_ms ? "MS" : "LR") << "\n";
          );
        }
        mode_used = (choose_ms ? "MS" : "LR");
        encodedBytes.push_back(choose_ms ? 1 : 0);
        if (!selected.empty()) {
          encodedBytes.insert(encodedBytes.end(), selected.begin(), selected.end());
        } else if (choose_ms) {
          std::vector<uint8_t> msBytes = encode_ms(start, block.size);
          encodedBytes.insert(encodedBytes.end(), msBytes.begin(), msBytes.end());
        } else {
          std::vector<uint8_t> lrBytes = encode_lr(start, block.size);
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

    size_t hardware_threads = std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t thread_limit = LAC::resolve_thread_limit(this->thread_count);
    if (thread_limit > 0) {
      hardware_threads = std::min(hardware_threads, thread_limit);
    }
    const size_t worker_count = std::min(hardware_threads, blocks.size());
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);

    for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back([&]() {
        try {
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

            encodedBlocks[block_idx] = encode_block(block_idx);
          }
        } catch (...) {
          stop_requested.store(true, std::memory_order_release);
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!worker_error) {
            worker_error = std::current_exception();
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

    writer.write_bits(static_cast<uint32_t>(blocks.size()), 32);
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (encodedBlocks[i].empty() ||
          encodedBlocks[i].size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("encoded block size is outside format limits");
      }
      writer.write_bits(blocks[i].size, 32);
      writer.write_bits(static_cast<uint32_t>(encodedBlocks[i].size()), 32);
    }

    size_t encoded_size = writer.get_buffer().size();
    for (const auto& bytes : encodedBlocks) {
      encoded_size += bytes.size();
    }
    writer.reserve_bytes(encoded_size);
    for (const auto& bytes : encodedBlocks) {
      writer.write_bytes(bytes.data(), bytes.size());
    }

    writer.flush_to_byte();
    return writer.take_buffer();
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

  void Encoder::set_thread_count(size_t max_threads) {
    this->thread_count = max_threads;
  }

} // namespace LAC
