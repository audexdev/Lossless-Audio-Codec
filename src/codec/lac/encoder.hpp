#pragma once
#include <vector>
#include <cstdint>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/block/encoder.hpp"
#include "codec/frame/frame_header.hpp"

namespace LAC {

class Encoder {
public:
    Encoder(uint16_t block_size, uint8_t order);

    std::vector<uint8_t> encode(const std::vector<int32_t>& left, const std::vector<int32_t>& right);

private:
    uint16_t block_size;
    uint8_t order;
};

} // namespace LAC