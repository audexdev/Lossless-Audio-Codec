#include "encoder.hpp"

namespace LAC {

Encoder::Encoder(uint16_t block_size, uint8_t order)
    : block_size(block_size), order(order) {}

std::vector<uint8_t> Encoder::encode(
    const std::vector<int32_t>& left,
    const std::vector<int32_t>& right
) {
    BitWriter writer;

    FrameHeader hdr;
    hdr.channels = (right.empty() ? 1 : 2);
    hdr.block_size = this->block_size;
    hdr.write(writer);

    Block::Encoder blockEnc(this->block_size, this->order);

    const size_t totalSamples = left.size();
    size_t pos = 0;

    while (pos < totalSamples) {
        size_t end = pos + this->block_size;
        if (end > totalSamples) end = totalSamples;

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

        pos = end;
    }

    writer.flush_to_byte();
    return writer.get_buffer();
}

} // namespace LAC