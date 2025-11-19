#pragma once
#include <vector>
#include <cstdint>
#include "codec/block/constants.hpp"
#include "codec/bitstream/bit_reader.hpp"

namespace Block {

class Decoder {
public:
    Decoder();

    bool decode(BitReader& br, uint32_t block_size, std::vector<int32_t>& out);
};

} // namespace Block
