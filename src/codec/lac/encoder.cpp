#include "encoder.hpp"
#include <limits>
#include <future>
#include <thread>
#include "codec/simd/neon.hpp"

namespace {
constexpr double kAbsDiffWeight = 1.0;
constexpr double kVarianceWeight = 0.5;
constexpr double kSlopePenaltyWeight = 0.25;
constexpr double kSlopeThreshold = 4.0;

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
}

namespace LAC {

Encoder::Encoder(uint8_t order, uint8_t stereo_mode, uint32_t sample_rate, uint8_t bit_depth, bool debug_lpc)
    : order(order),
      stereo_mode(stereo_mode),
      sample_rate(sample_rate),
      bit_depth(bit_depth),
      debug_lpc(debug_lpc),
      candidates{256, 512, 1024, 2048, 4096, 8192, 16384} {}

std::vector<uint8_t> Encoder::encode(
    const std::vector<int32_t>& left,
    const std::vector<int32_t>& right,
    ThreadCollector* collector
) {
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
    const bool useMidSide = isStereo && (hdr.stereo_mode == 1);

    std::vector<std::future<std::vector<uint8_t>>> blockTasks;
    blockTasks.reserve(blocks.size());

    auto launch_policy = std::thread::hardware_concurrency() > 1
        ? std::launch::async
        : std::launch::deferred;

    for (const auto& block : blocks) {
        blockTasks.emplace_back(std::async(
            launch_policy,
            [this, &left, &right, isStereo, useMidSide, block, collector]() -> std::vector<uint8_t> {
                Block::Encoder blockEnc(this->order, this->debug_lpc);
                std::vector<uint8_t> encodedBytes;
                const size_t start = block.start;
                const size_t end = start + block.size;

                if (collector) {
                    collector->record(std::this_thread::get_id());
                }

                if (block.size == 0) {
                    return encodedBytes;
                }

                if (useMidSide) {
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
                    encodedBytes.reserve(encodedMid.size() + encodedSide.size());
                    encodedBytes.insert(encodedBytes.end(), encodedMid.begin(), encodedMid.end());
                    encodedBytes.insert(encodedBytes.end(), encodedSide.begin(), encodedSide.end());
                } else {
                    std::vector<int32_t> blockL(left.begin() + start, left.begin() + end);
                    std::vector<uint8_t> encodedL = blockEnc.encode(blockL);
                    encodedBytes.insert(encodedBytes.end(), encodedL.begin(), encodedL.end());

                    if (isStereo) {
                        std::vector<int32_t> blockR(right.begin() + start, right.begin() + end);
                        std::vector<uint8_t> encodedR = blockEnc.encode(blockR);
                        encodedBytes.insert(encodedBytes.end(), encodedR.begin(), encodedR.end());
                    }
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
