#include "encoder.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "codec/bitstream/bit_writer.hpp"

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

    std::vector<int32_t> residual;
    lpc.compute_residual_q15(pcm, coeffs_q15, residual);

    int k = this->choose_rice_k(residual);
    uint32_t current_k = static_cast<uint32_t>(k);
    uint64_t sumU = 0;
    uint32_t count = 0;

    Rice rice;

    BitWriter bw;

    bw.write_bits(this->order, 8);
    bw.write_bits(k, 5);

    for (int i = 1; i <= this->order; ++i) {
        bw.write_bits(uint16_t(coeffs_q15[i]), 16);
    }

    for (int32_t r : residual) {
        rice.encode(bw, r, current_k);
        uint32_t u = static_cast<uint32_t>((r << 1) ^ (r >> 31));
        sumU += u;
        ++count;
        current_k = Rice::adapt_k(sumU, count);
    }

    bw.flush_to_byte();
    return bw.get_buffer();
}

} // namespace Block
