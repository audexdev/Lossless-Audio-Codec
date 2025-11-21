#include "codec/block/encoder.hpp"
#include "codec/block/decoder.hpp"
#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::vector<int32_t> make_random_block(size_t size, int32_t amplitude) {
    std::vector<int32_t> out(size);
    uint32_t state = 1;
    for (size_t i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        int32_t v = static_cast<int32_t>(state >> 5);
        out[i] = static_cast<int32_t>(v % amplitude);
    }
    return out;
}

std::vector<int32_t> make_sparse_block(size_t size) {
    std::vector<int32_t> out(size, 0);
    for (size_t i = 0; i < size; ++i) {
        if (i % 11 == 0) out[i] = 1;
        if (i % 53 == 0) out[i] = -1;
    }
    return out;
}

void roundtrip_block(const std::vector<int32_t>& pcm, bool enable_zr) {
    Block::Encoder enc(8);
    enc.set_zero_run_enabled(enable_zr);
    std::vector<uint8_t> buf = enc.encode(pcm);

    BitReader br(buf);
    Block::Decoder dec;
    std::vector<int32_t> decoded;
    if (!dec.decode(br, static_cast<uint32_t>(pcm.size()), decoded)) {
        std::cerr << "block decoder failed (size=" << pcm.size() << ")" << std::endl;
        std::cerr.flush();
        if (!buf.empty()) {
            BitReader dbg(buf);
            uint32_t mode = dbg.read_bits(8);
            int order = dbg.read_bits(8);
            int k0 = dbg.read_bits(5);
            std::cerr << "dbg mode=" << mode << " order=" << order << " k0=" << k0 << std::endl;
#ifdef LAC_ENABLE_ZERORUN
            auto read_rice_unsigned = [&](uint32_t k) -> uint32_t {
                uint32_t q = 0;
                while (dbg.read_bit() == 1u) {
                    ++q;
                }
                uint32_t rem = (k > 0) ? dbg.read_bits(k) : 0u;
                return (q << k) | rem;
            };
            uint32_t current_k = static_cast<uint32_t>(k0);
            dbg.read_bits(order * 16);
            uint32_t tag = dbg.read_bits(2);
            if (mode == 1 && !dbg.has_error()) {
                if (tag == 1) {
                    uint32_t run_delta = read_rice_unsigned(Block::ZERO_RUN_LENGTH_K);
                    std::cerr << "dbg first run delta=" << run_delta << " run_len=" << (run_delta + Block::ZERO_RUN_MIN_LENGTH) << std::endl;
                } else if (tag == 0) {
                    uint32_t u = read_rice_unsigned(current_k);
                    int32_t v = static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
                    std::cerr << "dbg first normal val=" << v << std::endl;
                } else {
                    std::cerr << "dbg unexpected tag=" << tag << std::endl;
                }
            }
#endif
        }
        assert(false);
    }
    if (decoded != pcm) {
        size_t mismatch = 0;
        while (mismatch < pcm.size() && mismatch < decoded.size() && decoded[mismatch] == pcm[mismatch]) {
            ++mismatch;
        }
        uint8_t mode = buf.empty() ? 0 : buf[0];
        std::cerr << "roundtrip mismatch at " << mismatch
                  << " expected=" << pcm[mismatch]
                  << " got=" << decoded[mismatch]
                  << " mode_byte=" << static_cast<unsigned int>(mode)
                  << " encoded_bytes=" << buf.size()
                  << " decoded_size=" << decoded.size()
                  << " first_bytes="
                  << std::hex
                  << static_cast<int>(buf.size() > 0 ? buf[0] : 0) << " "
                  << static_cast<int>(buf.size() > 1 ? buf[1] : 0) << " "
                  << static_cast<int>(buf.size() > 2 ? buf[2] : 0) << " "
                  << static_cast<int>(buf.size() > 3 ? buf[3] : 0)
                  << std::dec
                  << "\n";
        std::cerr << "decoded head: ";
        for (size_t i = 0; i < std::min<size_t>(5, decoded.size()); ++i) {
            std::cerr << decoded[i] << " ";
        }
        std::cerr << "\n";
        if (mode == 1u) {
            BitReader dbg(buf);
            uint32_t m = dbg.read_bits(8);
            int order = dbg.read_bits(8);
            int k0 = dbg.read_bits(5);
            std::vector<int16_t> coeffs(order + 1);
            coeffs[0] = 0;
            for (int i = 1; i <= order; ++i) {
                coeffs[i] = static_cast<int16_t>(dbg.read_bits(16));
            }
            auto read_rice_unsigned = [&](uint32_t k) -> uint32_t {
                uint32_t q = 0;
                while (dbg.read_bit() == 1u) {
                    ++q;
                }
                uint32_t rem = (k > 0) ? dbg.read_bits(k) : 0u;
                return (q << k) | rem;
            };
            auto zigzag_decode = [](uint32_t u) -> int32_t {
                return static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
            };
            std::vector<int32_t> residual(pcm.size(), 0);
            uint32_t idx = 0;
            uint32_t current_k = static_cast<uint32_t>(k0);
            uint64_t sum_u = 0;
            bool would_fail = false;
            uint32_t norm_tokens = 0, run_tokens = 0, esc_tokens = 0;
            int32_t first_norm_val = 0;
            int32_t first_escape_val = 0;
            while (idx < residual.size() && !dbg.has_error()) {
                uint32_t tag = dbg.read_bits(2);
                if (tag == 0) {
                    if (idx >= residual.size()) {
                        would_fail = true;
                        break;
                    }
                    ++norm_tokens;
                    uint32_t u = read_rice_unsigned(current_k);
                    residual[idx++] = zigzag_decode(u);
                    if (norm_tokens == 1) {
                        first_norm_val = residual[idx - 1];
                    }
                    sum_u += u;
                    current_k = Rice::adapt_k(sum_u, idx);
                } else if (tag == 1) {
                    uint32_t run = read_rice_unsigned(Block::ZERO_RUN_LENGTH_K) + Block::ZERO_RUN_MIN_LENGTH;
                    if (idx + run > residual.size()) {
                        would_fail = true;
                        break;
                    }
                    ++run_tokens;
                    for (uint32_t j = 0; j < run && idx < residual.size(); ++j) {
                        residual[idx++] = 0;
                        current_k = Rice::adapt_k(sum_u, idx);
                    }
                } else if (tag == 2) {
                    if (idx >= residual.size()) {
                        would_fail = true;
                        break;
                    }
                    uint32_t zz = dbg.read_bits(32);
                    if (dbg.has_error()) {
                        would_fail = true;
                        break;
                    }
                    ++esc_tokens;
                    if (esc_tokens == 1) {
                        first_escape_val = zigzag_decode(zz);
                    }
                    residual[idx++] = zigzag_decode(zz);
                    uint32_t u = static_cast<uint32_t>((residual[idx - 1] << 1) ^ (residual[idx - 1] >> 31));
                    sum_u += u;
                    current_k = Rice::adapt_k(sum_u, idx);
                } else {
                    would_fail = true;
                    break;
                }
            }
            std::vector<int32_t> pcm_dbg;
            LPC lpc(order);
            lpc.restore_from_residual_q15(residual, coeffs, pcm_dbg);
            std::cerr << "manual decode mode=" << m << " k0=" << k0
                      << " residual_first=" << residual[0]
                      << " pcm_first=" << (pcm_dbg.empty() ? 0 : pcm_dbg[0])
                      << " would_fail=" << would_fail
                      << " has_error=" << dbg.has_error()
                      << " idx=" << idx
                      << " norm_tokens=" << norm_tokens
                      << " run_tokens=" << run_tokens
                      << " esc_tokens=" << esc_tokens
                      << " first_norm_val=" << first_norm_val
                      << " first_escape=" << first_escape_val
                      << "\n";
        }
        assert(false);
    }
}

void roundtrip_lac(const std::vector<int32_t>& left,
                   const std::vector<int32_t>& right) {
    LAC::Encoder enc(12, 0, 44100, 16, false, false, false);
    std::vector<uint8_t> bs = enc.encode(left, right);
    LAC::Decoder dec;
    std::vector<int32_t> outL, outR;
    FrameHeader hdr;
    assert(dec.decode(bs.data(), bs.size(), outL, outR, &hdr));
    assert(outL == left);
    assert(outR == right);
}

} // namespace

void run_zerorun_tests() {
    {
        std::vector<int32_t> zeros(Block::MAX_BLOCK_SIZE / 4, 0);
        roundtrip_block(zeros, true);
        std::cout << "zero-run dense block ok\n";
    }

    {
        // Mixed segment with a long run that should collapse.
        std::vector<int32_t> pcm = {5, 4, 3, 2, 1, 0, 0, 0, 0, 7, 6, 0, 0, 0, 0, 0, -3};
        roundtrip_block(pcm, true);
        std::cout << "mixed run block ok\n";
    }

    {
        // Escape-heavy values.
        std::vector<int32_t> spikes(128);
        for (size_t i = 0; i < spikes.size(); ++i) {
            spikes[i] = (i & 1) ? (1 << 23) : -(1 << 23);
        }
        roundtrip_block(spikes, true);
        std::cout << "escape-heavy block ok\n";
    }

    {
        std::vector<int32_t> short_runs(256);
        for (size_t i = 0; i < short_runs.size(); ++i) {
            short_runs[i] = (i % 5 == 0) ? 0 : static_cast<int32_t>((i & 1) ? 12 : -12);
        }
        roundtrip_block(short_runs, true);
        std::cout << "short run mix ok\n";
    }

    {
        std::vector<int32_t> alternation(512);
        for (size_t i = 0; i < alternation.size(); ++i) {
            alternation[i] = (i & 1) ? 0 : static_cast<int32_t>(i % 4);
        }
        roundtrip_block(alternation, true);
        std::cout << "alternating zeros ok\n";
    }

    {
        std::vector<int32_t> long_run(1024, 0);
        for (size_t i = 200; i < 220; ++i) long_run[i] = 3;
        roundtrip_block(long_run, true);
        std::cout << "long run block ok\n";
    }

    {
        // Backward-compatible path: no residual_mode byte.
        auto pcm = make_random_block(400, 8000);
        Block::Encoder enc(8);
        enc.set_zero_run_enabled(false);
        std::vector<uint8_t> buf = enc.encode(pcm);
        BitReader br(buf);
        Block::Decoder dec;
        std::vector<int32_t> decoded;
        assert(dec.decode(br, static_cast<uint32_t>(pcm.size()), decoded));
        assert(decoded == pcm);
        std::cout << "legacy residual decode ok\n";
    }

    {
        roundtrip_block(make_sparse_block(1024), true);
        std::cout << "sparse block ok\n";
    }

    {
        roundtrip_block(make_random_block(2048, 1 << 23), true);
        std::cout << "high-entropy noise ok\n";
    }

    {
        // Random sweep.
        for (size_t i = 64; i < 640; i += 64) {
            roundtrip_block(make_random_block(i, 12000), true);
        }
        std::cout << "random block sweep ok\n";
    }

    {
        // LR/MS roundtrip via full LAC path.
        std::vector<int32_t> left = make_sparse_block(512);
        std::vector<int32_t> right = make_sparse_block(512);
        roundtrip_lac(left, right);
        std::cout << "lac lr/ms roundtrip ok\n";
    }

    {
        // Compression benefit check.
        std::vector<int32_t> pcm(Block::MAX_BLOCK_SIZE, 0);
        Block::Encoder enc_zr(8);
        enc_zr.set_zero_run_enabled(true);
        std::vector<uint8_t> zr_buf = enc_zr.encode(pcm);

        Block::Encoder enc_plain(8);
        enc_plain.set_zero_run_enabled(false);
        std::vector<uint8_t> plain_buf = enc_plain.encode(pcm);

        assert(zr_buf.size() < plain_buf.size());
        std::cout << "compression comparison ok\n";
    }
}
