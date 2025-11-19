#include "encoder.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "codec/bitstream/bit_writer.hpp"
#include "codec/simd/neon.hpp"
#include "codec/simd/rice_neon.hpp"

namespace Block {

Encoder::Encoder(int order) : order(order) {}

int Encoder::choose_rice_k(const std::vector<int32_t>& residual) {
    int64_t sum = 0;
    for (int32_t r : residual) sum += (r >= 0 ? r : -r);

    double avg = double(sum) / double(residual.size());
    int k = 0;
    while ((1 << k) < avg && k < 15) k++;
    return k;
}

std::vector<uint8_t> Encoder::encode(const std::vector<int32_t>& pcm) {
    LPC lpc(this->order);

    std::vector<int16_t> coeffs_q15;
    lpc.analyze_block_q15(pcm, coeffs_q15);

    std::vector<int32_t> residual(pcm.size());
    if (!pcm.empty() && !coeffs_q15.empty()) {
        SIMD::lpc_residual_simd_or_scalar(pcm.data(),
                                          coeffs_q15.data(),
                                          residual.data(),
                                          pcm.size(),
                                          this->order);
    }

    std::vector<uint32_t> rice_unsigned(residual.size());
    std::vector<uint64_t> rice_prefix(residual.size());
    if (!residual.empty()) {
        SIMD::rice_unsigned_simd_or_scalar(residual.data(),
                                           rice_unsigned.data(),
                                           residual.size());
        SIMD::rice_prefix_sum_simd_or_scalar(rice_unsigned.data(),
                                             rice_prefix.data(),
                                             residual.size());
    }

    std::vector<uint32_t> rice_k_next(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        rice_k_next[i] = Rice::adapt_k(rice_prefix[i],
                                       static_cast<uint32_t>(i + 1));
    }

    int k = this->choose_rice_k(residual);
    uint32_t current_k = static_cast<uint32_t>(k);

    Rice rice;

    BitWriter bw;

    bw.write_bits(this->order, 8);
    bw.write_bits(k, 5);

    for (int i = 1; i <= this->order; ++i) {
        bw.write_bits(uint16_t(coeffs_q15[i]), 16);
    }

    for (size_t i = 0; i < residual.size(); ++i) {
        rice.encode(bw, residual[i], current_k);
        current_k = rice_k_next[i];
    }

    bw.flush_to_byte();
    return bw.get_buffer();
}

} // namespace Block
