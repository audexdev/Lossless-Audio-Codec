#include "encoder.hpp"
#include <limits>
#include <future>
#include <stdexcept>
#include <string>
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
      sum_u += static_cast<uint32_t>((r << 1) ^ (r >> 31));
    }
    const uint64_t mean = (sum_u + (residual.size() >> 1)) / residual.size();
    uint32_t k = 0;
    while ((1u << k) < mean && k < 31u) ++k;

    uint64_t bits = 0;
    for (int32_t r : residual) {
      const uint32_t u = static_cast<uint32_t>((r << 1) ^ (r >> 31));
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
      uint64_t& bits) {
    LPC lpc(order);
    std::vector<int16_t> coeffs;
    int used_order = 0;
    long double energy = 0.0L;
    if (!lpc.analyze_block_q15(pcm, coeffs, used_order, &energy)) {
      return false;
    }
    if (used_order <= 0) return false;

    std::vector<int32_t> residual(pcm.size());
    lpc.compute_residual_q15(pcm, coeffs, residual);
    bits = estimate_rice_bits(residual);
    return true;
  }

  StereoCost estimate_stereo_cost(const std::vector<int32_t>& left,
      const std::vector<int32_t>& right,
      size_t start,
      size_t size,
      int order) {
    StereoCost cost;
    if (size == 0 || start + size > left.size() || start + size > right.size()) {
      return cost;
    }

    std::vector<int32_t> blockL(left.begin() + start, left.begin() + start + size);
    std::vector<int32_t> blockR(right.begin() + start, right.begin() + start + size);
    cost.lr_valid = estimate_channel_cost(blockL, order, cost.lr_bits);
    uint64_t bits_r = 0;
    if (estimate_channel_cost(blockR, order, bits_r)) {
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
    cost.ms_valid = estimate_channel_cost(blockM, order, cost.ms_bits);
    uint64_t bits_s = 0;
    if (estimate_channel_cost(blockS, order, bits_s)) {
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

    if (this->zero_run_enabled && this->sample_rate >= 96000 && !blocks.empty()) {
      const size_t guard_blocks = std::min<size_t>(blocks.size(), 4);
      Block::Encoder estimator(this->order, this->debug_lpc, this->debug_zr);
      estimator.set_zero_run_enabled(true);
      uint64_t normal_bits = 0;
      uint64_t zr_bits = 0;
      uint64_t bin_bits = 0;
      for (size_t i = 0; i < guard_blocks; ++i) {
        const auto& blk = blocks[i];
        std::vector<int32_t> blockL(left.begin() + blk.start, left.begin() + blk.start + blk.size);
        uint64_t bn = 0, bz = 0, bb = 0;
        if (estimator.estimate_bits(blockL, bn, bz, bb)) {
          normal_bits += bn;
          zr_bits += bz;
          bin_bits += bb;
        }
        if (isStereo) {
          std::vector<int32_t> blockR(right.begin() + blk.start, right.begin() + blk.start + blk.size);
          if (estimator.estimate_bits(blockR, bn, bz, bb)) {
            normal_bits += bn;
            zr_bits += bz;
            bin_bits += bb;
          }
        }
      }
      if (normal_bits > 0) {
        double saving = 1.0 - (static_cast<double>(zr_bits) / static_cast<double>(normal_bits));
        if (saving < 0.02) {
          this->zero_run_enabled = false;
        }
      }
    }

    std::vector<std::future<std::vector<uint8_t>>> blockTasks;
    blockTasks.reserve(blocks.size());

    auto launch_policy = std::thread::hardware_concurrency() > 1
      ? std::launch::async
      : std::launch::deferred;

    for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
      const auto& block = blocks[block_idx];
      blockTasks.emplace_back(std::async(
            launch_policy,
            [this, &left, &right, isStereo, forceMidSide, perBlockStereo, globalStereoMode, block, block_idx, collector]() -> std::vector<uint8_t> {
            Block::Encoder blockEnc(this->order, this->debug_lpc, this->debug_zr);
            blockEnc.set_zero_run_enabled(this->zero_run_enabled);
            blockEnc.set_partitioning_enabled(this->partitioning_enabled);
            blockEnc.set_debug_partitions(this->debug_partitions);
            blockEnc.set_debug_block_index(block_idx);
            std::vector<uint8_t> encodedBytes;
            const size_t start = block.start;
            const size_t end = start + block.size;

            if (collector) {
            collector->record(std::this_thread::get_id());
            }

            if (block.size == 0) {
            return encodedBytes;
            }

            auto encode_lr = [&]() -> std::vector<uint8_t> {
              std::vector<uint8_t> out;
              std::vector<int32_t> blockL(left.begin() + start, left.begin() + end);
              std::vector<uint8_t> encodedL = blockEnc.encode(blockL);
              out.insert(out.end(), encodedL.begin(), encodedL.end());
              if (isStereo) {
                std::vector<int32_t> blockR(right.begin() + start, right.begin() + end);
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
              std::vector<uint8_t> encodedMid = blockEnc.encode(blockMid);
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
              StereoCost cost = estimate_stereo_cost(left, right, start, block.size, this->order);
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
            }
      ));
    }

    for (auto& futureBytes : blockTasks) {
      std::vector<uint8_t> bytes = futureBytes.get();
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
    __int128 sumsq = 0;

    const size_t end = position + length;
    const int32_t first = channel[position];
    sum = static_cast<int64_t>(first);
    sumsq = static_cast<__int128>(first) * static_cast<__int128>(first);

    if constexpr (SIMD::kHasNeon) {
      absdiff_sum = SIMD::neon_absdiff_sum(channel.data() + position, length);
      for (size_t i = position + 1; i < end; ++i) {
        const int64_t v = static_cast<int64_t>(channel[i]);
        sum += v;
        sumsq += static_cast<__int128>(v) * static_cast<__int128>(v);
      }
    } else {
      int32_t prev = first;
      for (size_t i = position + 1; i < end; ++i) {
        const int32_t cur = channel[i];
        const int64_t v = static_cast<int64_t>(cur);
        sum += v;
        sumsq += static_cast<__int128>(v) * static_cast<__int128>(v);
        const int64_t diff = static_cast<int64_t>(cur) - static_cast<int64_t>(prev);
        absdiff_sum += (diff >= 0 ? diff : -diff);
        prev = cur;
      }
    }

    const double absdiff_avg = static_cast<double>(absdiff_sum) / static_cast<double>(length);
    const double mean = static_cast<double>(sum) / static_cast<double>(length);
    const double mean_sq = mean * mean;
    const double avg_sq = static_cast<double>(sumsq) / static_cast<double>(length);
    double variance = avg_sq - mean_sq;
    if (variance < 0.0) variance = 0.0;

    return (kAbsDiffWeight * absdiff_avg) + (kVarianceWeight * variance);
  }

} // namespace LAC
