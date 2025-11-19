#pragma once
#include <vector>
#include <cstdint>
#include "codec/block/constants.hpp"

namespace Block {

class Encoder {
public:
    explicit Encoder(int order);

    // input: int32_t PCM block
    // output: compressed block as bytes
    std::vector<uint8_t> encode(const std::vector<int32_t>& pcm);

private:
    int order;

    int choose_rice_k(const std::vector<int32_t>& residual);
};

} // namespace Block
