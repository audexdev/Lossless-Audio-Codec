#pragma once
#include <cstdint>
#include <vector>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/bitstream/bit_reader.hpp"

class Rice {
public:
    static void encode(BitWriter& w, int32_t value, uint32_t k);

    static int32_t decode(BitReader& r, uint32_t k);

    // Compute optimal k from residual block
    static uint32_t compute_k(const std::vector<int32_t>& residuals);

    static uint32_t adapt_k(uint64_t sum, uint32_t count);

private:
    static uint32_t signed_to_unsigned(int32_t v);
    static int32_t unsigned_to_signed(uint32_t u);
};
