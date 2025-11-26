#include "codec/block/encoder.hpp"
#include "codec/block/decoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

uint8_t read_predictor_type(const std::vector<uint8_t>& buf) {
    if (buf.empty()) return 2;
    BitReader br(buf);
    return static_cast<uint8_t>(br.read_bits(8));
}

std::vector<int32_t> make_ramp(size_t n) {
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<int32_t>(i * 16);
    }
    return out;
}

std::vector<int32_t> make_fir_sequence(size_t n) {
    std::vector<int32_t> out(n);
    if (n == 0) return out;
    out[0] = 100000;
    if (n > 1) out[1] = 80000;
    for (size_t i = 2; i < n; ++i) {
        int64_t pred = (3LL * out[i - 1] - out[i - 2]) >> 2;
        out[i] = static_cast<int32_t>(pred);
    }
    return out;
}

std::vector<int32_t> make_noise(size_t n) {
    std::vector<int32_t> out(n);
    uint32_t state = 1;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = static_cast<int32_t>((state >> 8) & 0xFFFF);
    }
    return out;
}

void assert_roundtrip(const std::vector<int32_t>& pcm, const std::vector<uint8_t>& expected) {
    Block::Encoder enc(12);
    enc.set_partitioning_enabled(false);
    enc.set_zero_run_enabled(false);
    std::vector<uint8_t> buf = enc.encode(pcm);
    uint8_t predictor = read_predictor_type(buf);
    bool ok_type = false;
    for (uint8_t e : expected) {
        if (predictor == e) ok_type = true;
    }
    assert(ok_type);

    BitReader br(buf);
    Block::Decoder dec;
    std::vector<int32_t> decoded;
    assert(dec.decode(br, static_cast<uint32_t>(pcm.size()), decoded));
    assert(decoded == pcm);
}

} // namespace

void run_predictor_tests() {
    assert_roundtrip(make_ramp(128), {0, 2});           // fixed or LPC
    assert_roundtrip(make_fir_sequence(256), {1, 2});   // FIR or LPC
    assert_roundtrip(make_noise(256), {2});             // LPC preferred
    std::cout << "predictor selection tests ok\n";
}
