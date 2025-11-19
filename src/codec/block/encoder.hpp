#pragma once
#include <vector>
#include <cstdint>

namespace Block {

class Encoder {
public:
    Encoder(int block_size, int order);

    // input: int32_t PCM block (size = block_size)
    // output: compressed block as bytes
    std::vector<uint8_t> encode(const std::vector<int32_t>& pcm);

private:
    int block_size;
    int order;

    int choose_rice_k(const std::vector<int32_t>& residual);
};

} // namespace Block