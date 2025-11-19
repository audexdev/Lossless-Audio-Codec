#pragma once
#include <vector>
#include <cstdint>
#include "codec/bitstream/bit_reader.hpp"

namespace Block {

class Decoder {
public:
    Decoder();

    std::vector<int32_t> decode(BitReader& br);
};

} // namespace Block