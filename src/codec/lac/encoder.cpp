#include "encoder.hpp"
#include <limits>

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
    const std::vector<int32_t>& right
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

    Block::Encoder blockEnc(this->order);

    for (const auto& block : blocks) {
        size_t start = block.start;
        size_t end = start + block.size;

        if (hdr.channels == 2 && hdr.stereo_mode == 1) {
            std::vector<int32_t> blockMid(block.size);
            std::vector<int32_t> blockSide(block.size);
            for (size_t i = 0; i < block.size; ++i) {
                int32_t l = left[start + i];
                int32_t r = right[start + i];
                blockMid[i] = (l + r) >> 1;
                blockSide[i] = l - r;
            }

            std::vector<uint8_t> encodedMid = blockEnc.encode(blockMid);
            for (uint8_t b : encodedMid)
                writer.write_bits(b, 8);

            std::vector<uint8_t> encodedSide = blockEnc.encode(blockSide);
            for (uint8_t b : encodedSide)
                writer.write_bits(b, 8);
        } else {
            std::vector<int32_t> blockL(left.begin() + start, left.begin() + end);
            std::vector<uint8_t> encodedL = blockEnc.encode(blockL);

            for (uint8_t b : encodedL)
                writer.write_bits(b, 8);

            if (hdr.channels == 2) {
                std::vector<int32_t> blockR(right.begin() + start, right.begin() + end);
                std::vector<uint8_t> encodedR = blockEnc.encode(blockR);

                for (uint8_t b : encodedR)
                    writer.write_bits(b, 8);
            }
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
    int32_t prev = channel[position];
    for (size_t i = 1; i < length; ++i) {
        int32_t cur = channel[position + i];
        int64_t diff = static_cast<int64_t>(cur) - static_cast<int64_t>(prev);
        sum += (diff >= 0 ? diff : -diff);
        prev = cur;
    }

    return static_cast<double>(sum) / static_cast<double>(length);
}

} // namespace LAC
