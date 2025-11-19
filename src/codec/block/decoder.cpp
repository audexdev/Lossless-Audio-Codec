#include "decoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"

namespace Block {

Decoder::Decoder() {}

std::vector<int32_t> Decoder::decode(BitReader& br) {
    int block_size = br.read_bits(16);
    int order = br.read_bits(8);
    int k = br.read_bits(5);
    uint32_t current_k = static_cast<uint32_t>(k);
    uint64_t sumU = 0;
    uint32_t count = 0;

    std::vector<int16_t> coeffs_q15(order + 1);
    coeffs_q15[0] = 0;

    for (int i = 1; i <= order; ++i) {
        coeffs_q15[i] = int16_t(br.read_bits(16));
    }

    Rice rice;
    std::vector<int32_t> residual(block_size);
    for (int i = 0; i < block_size; ++i) {
        residual[i] = rice.decode(br, current_k);
        uint32_t u = static_cast<uint32_t>((residual[i] << 1) ^ (residual[i] >> 31));
        sumU += u;
        ++count;
        current_k = Rice::adapt_k(sumU, count);
    }

    br.align_to_byte();

    LPC lpc(order);

    std::vector<int32_t> pcm;
    lpc.restore_from_residual_q15(residual, coeffs_q15, pcm);

    return pcm;
}

} // namespace Block
