#include "rice.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

uint32_t Rice::signed_to_unsigned(int32_t v) {
  const uint32_t sign_mask = (v < 0) ? std::numeric_limits<uint32_t>::max() : 0u;
  return (static_cast<uint32_t>(v) << 1) ^ sign_mask;
}

int32_t Rice::unsigned_to_signed(uint32_t u) {
    if ((u & 1u) == 0u) return static_cast<int32_t>(u >> 1);
    return static_cast<int32_t>(-static_cast<int64_t>((u >> 1) + 1u));
}

void Rice::encode(BitWriter& w, int32_t value, uint32_t k) {
    uint32_t u = signed_to_unsigned(value);

    uint32_t q = u >> k;
    uint32_t r = u & ((1u << k) - 1);

    w.write_unary_ones(q);
    w.write_bit(0);

    if (k > 0) {
        w.write_bits(r, k);
    }
}

bool Rice::decode(BitReader& r, uint32_t k, int32_t& value) {
    if (k > 31u) return false;

    uint32_t q = 0;
    const uint32_t max_q = std::numeric_limits<uint32_t>::max() >> k;
    if (!r.read_unary_ones(max_q, q)) return false;

    uint32_t u = q << k;

    uint32_t rem = 0;
    if (k > 0) {
        rem = r.read_bits(k);
        if (r.has_error()) return false;
    }

    u |= rem;
    value = unsigned_to_signed(u);
    return true;
}
