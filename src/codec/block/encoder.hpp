#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include "codec/block/constants.hpp"

namespace Block {

class Encoder {
public:
    explicit Encoder(int order, bool debug_lpc = false, bool debug_zr = false);

    // input: int32_t PCM block
    // output: compressed block as bytes
    std::vector<uint8_t> encode(const std::vector<int32_t>& pcm);

    void set_zero_run_enabled(bool enabled);
    void set_debug_block_index(size_t index);
    void set_partitioning_enabled(bool enabled);
    void set_debug_partitions(bool enabled);
    bool estimate_bits(const std::vector<int32_t>& pcm,
                       uint64_t& bits_normal,
                       uint64_t& bits_zr,
                       uint64_t& bits_bin);

private:
    int order;
    bool debug_lpc;
    bool debug_zr;
    bool zero_run_enabled = false;
    bool partitioning_enabled = false;
    bool debug_partitions = false;
    size_t block_index = 0;

    int choose_rice_k(const std::vector<int32_t>& residual);
};

} // namespace Block
