#include "encoder.hpp"

namespace LAC {

Encoder::Encoder(uint16_t block_size, uint8_t order, uint8_t stereo_mode, uint32_t sample_rate, uint8_t bit_depth)
    : block_size(block_size), order(order), stereo_mode(stereo_mode), sample_rate(sample_rate), bit_depth(bit_depth) {}

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
    hdr.block_size = this->block_size;
    hdr.write(writer);

    Block::Encoder blockEnc(this->block_size, this->order);

    const size_t totalSamples = left.size();
    size_t pos = 0;

    while (pos < totalSamples) {
        size_t end = pos + this->block_size;
        if (end > totalSamples) end = totalSamples;

        if (hdr.channels == 2 && hdr.stereo_mode == 1) {
            std::vector<int32_t> blockMid(end - pos);
            std::vector<int32_t> blockSide(end - pos);
            for (size_t i = 0; i < end - pos; ++i) {
                int32_t l = left[pos + i];
                int32_t r = right[pos + i];
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
            std::vector<int32_t> blockL(left.begin() + pos, left.begin() + end);
            std::vector<uint8_t> encodedL = blockEnc.encode(blockL);

            for (uint8_t b : encodedL)
                writer.write_bits(b, 8);

            if (hdr.channels == 2) {
                std::vector<int32_t> blockR(right.begin() + pos, right.begin() + end);
                std::vector<uint8_t> encodedR = blockEnc.encode(blockR);

                for (uint8_t b : encodedR)
                    writer.write_bits(b, 8);
            }
        }

        pos = end;
    }

    writer.flush_to_byte();
    return writer.get_buffer();
}

} // namespace LAC
