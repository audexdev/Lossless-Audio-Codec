#pragma once
#include <vector>
#include <cstdint>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/block/encoder.hpp"
#include "codec/frame/frame_header.hpp"

namespace LAC {

class Encoder {
public:
    Encoder(uint16_t block_size, uint8_t order, uint8_t stereo_mode = 0, uint32_t sample_rate = 44100, uint8_t bit_depth = 16);

    std::vector<uint8_t> encode(const std::vector<int32_t>& left, const std::vector<int32_t>& right);

private:
    uint16_t block_size;
    uint8_t order;
    uint8_t stereo_mode;
    uint32_t sample_rate;
    uint8_t bit_depth;
};

} // namespace LAC
