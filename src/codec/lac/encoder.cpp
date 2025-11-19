#include "encoder.hpp"
#include <limits>
#include <future>
#include <thread>
#include "codec/simd/neon.hpp"

namespace {
struct BlockInfo {
    size_t start;
    uint32_t size;
};
}

namespace LAC {

Encoder::Encoder(uint8_t order, uint8_t stereo_mode, uint32_t sample_rate, uint8_t bit_depth)
    : order(order),
      stereo_mode(stereo_mode),
      sample_rate(sample_rate),
      bit_depth(bit_depth),
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
                Block::Encoder blockEnc(this->order);
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

    int64_t sum = 0;
    if constexpr (SIMD::kHasNeon) {
        sum = SIMD::neon_absdiff_sum(channel.data() + position, length);
    } else {
        int32_t prev = channel[position];
        for (size_t i = 1; i < length; ++i) {
            int32_t cur = channel[position + i];
            int64_t diff = static_cast<int64_t>(cur) - static_cast<int64_t>(prev);
            sum += (diff >= 0 ? diff : -diff);
            prev = cur;
        }
    }

    return static_cast<double>(sum) / static_cast<double>(length);
}

} // namespace LAC
