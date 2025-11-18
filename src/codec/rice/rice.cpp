#include "rice.hpp"
#include <cmath>

uint32_t Rice::signed_to_unsigned(int32_t v) {
    return (uint32_t)((v << 1) ^ (v >> 31));
}

int32_t Rice::unsigned_to_signed(uint32_t u) {
    return (u >> 1) ^ -((int32_t)(u & 1));
}

void Rice::encode(BitWriter& w, int32_t value, uint32_t k) {
    uint32_t u = signed_to_unsigned(value);

    uint32_t q = u >> k;
    uint32_t r = u & ((1u << k) - 1);

    for (uint32_t i = 0; i < q; ++i) {
        w.write_bit(1);
    }
    w.write_bit(0);

    if (k > 0) {
        w.write_bits(r, k);
    }
}

int32_t Rice::decode(BitReader& r, uint32_t k) {
    uint32_t q = 0;
    while (r.read_bit() == 1) {
        q++;
    }

    uint32_t u = q << k;

    uint32_t rem = 0;
    if (k > 0) {
        rem = r.read_bits(k);
    }

    u |= rem;

    return unsigned_to_signed(u);
}

uint32_t Rice::compute_k(const std::vector<int32_t>& residuals) {
    uint64_t sum = 0;
    for (int32_t v : residuals) {
        uint32_t u = signed_to_unsigned(v);
        sum += u;
    }

    if (residuals.empty()) return 0;

    double mean = double(sum) / double(residuals.size());

    uint32_t k = 0;
    while ((1u << k) < (uint32_t)(mean + 0.5) && k < 31) {
        k++;
    }

    return k;
}