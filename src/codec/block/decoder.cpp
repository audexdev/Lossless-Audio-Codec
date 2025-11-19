#include "decoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"

namespace Block {

Decoder::Decoder() {}

bool Decoder::decode(BitReader& br, uint32_t block_size, std::vector<int32_t>& out) {
    if (block_size == 0 || block_size > MAX_BLOCK_SIZE) return false;

    int order = br.read_bits(8);
    if (br.has_error()) return false;
    if (order <= 0 || order > 32 || static_cast<uint32_t>(order) >= block_size) return false;

    int k = br.read_bits(5);
    if (br.has_error()) return false;
    uint32_t current_k = static_cast<uint32_t>(k);
    uint64_t sumU = 0;
    uint32_t count = 0;

    std::vector<int16_t> coeffs_q15(order + 1);
    coeffs_q15[0] = 0;

    for (int i = 1; i <= order; ++i) {
        coeffs_q15[i] = int16_t(br.read_bits(16));
        if (br.has_error()) return false;
    }

    Rice rice;
    std::vector<int32_t> residual(block_size);
    for (uint32_t i = 0; i < block_size; ++i) {
        residual[i] = rice.decode(br, current_k);
        uint32_t u = static_cast<uint32_t>((residual[i] << 1) ^ (residual[i] >> 31));
        sumU += u;
        ++count;
        current_k = Rice::adapt_k(sumU, count);
        if (br.has_error()) return false;
    }

    br.align_to_byte();
    if (br.has_error()) return false;

    LPC lpc(order);

    std::vector<int32_t> pcm;
    lpc.restore_from_residual_q15(residual, coeffs_q15, pcm);
    if (pcm.size() != block_size) return false;
    out.swap(pcm);
    return true;
}

} // namespace Block
