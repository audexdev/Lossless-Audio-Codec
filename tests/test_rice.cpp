// Direct Rice coder tests: round-trip, signed-mapping boundaries, and the
// k > 31 rejection contract. Built with asserts forced on (-UNDEBUG) and run
// under the ASan/UBSan CI job to catch shift/overflow UB.
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "codec/bitstream/bit_reader.hpp"
#include "codec/bitstream/bit_writer.hpp"
#include "codec/rice/rice.hpp"

namespace {

bool roundtrip(int32_t value, uint32_t k) {
    BitWriter w;
    Rice::encode(w, value, k);
    w.flush_to_byte();
    std::vector<uint8_t> buf = w.take_buffer();
    BitReader r(buf.data(), buf.size());
    int32_t out = 0;
    if (!Rice::decode(r, k, out)) return false;
    return out == value;
}

} // namespace

int main() {
    // Moderate values round-trip for every valid k. The unsigned-mapped value
    // stays small, so the unary quotient is bounded even at k = 0.
    for (uint32_t k = 0; k <= 31; ++k) {
        for (int32_t v = -64; v <= 64; ++v) {
            assert(roundtrip(v, k));
        }
    }

    // Full-range boundary values, restricted to high k so the quotient is tiny.
    // INT32_MIN maps to the maximum unsigned value (0xFFFFFFFF), which is the
    // hardest case for the k == 31 shift boundary.
    const int32_t kMin = std::numeric_limits<int32_t>::min();
    const int32_t kMax = std::numeric_limits<int32_t>::max();
    for (uint32_t k = 24; k <= 31; ++k) {
        assert(roundtrip(0, k));
        assert(roundtrip(1, k));
        assert(roundtrip(-1, k));
        assert(roundtrip(kMax, k));
        assert(roundtrip(kMin, k));
    }
    assert(roundtrip(kMin, 31));
    assert(roundtrip(kMax, 31));

    // Decode must reject k > 31 (shifts by >= 32 are undefined behaviour).
    {
        const std::vector<uint8_t> data(8, 0xFFu);
        for (uint32_t k = 32; k <= 40; ++k) {
            BitReader r(data.data(), data.size());
            int32_t out = 0;
            assert(!Rice::decode(r, k, out));
        }
    }

    std::cout << "rice tests ok\n";
    return 0;
}
