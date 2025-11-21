#include "codec/block/constants.hpp"
#include "codec/block/decoder.hpp"
#include "codec/block/encoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/lac/decoder.hpp"
#include "codec/lac/encoder.hpp"
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

uint8_t read_partition_order(const std::vector<uint8_t>& buffer) {
    if (buffer.empty()) return 0;
    BitReader br(buffer);
    uint32_t first = br.read_bits(8);
    if ((first & Block::PARTITION_FLAG) != 0u) {
        return static_cast<uint8_t>((first & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT);
    }
    return 0;
}

std::vector<int32_t> make_composite_pattern(size_t total) {
    std::vector<int32_t> pcm(total, 0);
    size_t quarter = total / 4;
    size_t half = total / 2;
    const int32_t burst = 1 << 22;

    for (size_t i = quarter; i < half; ++i) {
        pcm[i] = static_cast<int32_t>((static_cast<int64_t>(i - quarter) * 4000) / static_cast<int64_t>(half - quarter + 1));
    }

    uint32_t state = 1;
    for (size_t i = half; i < half + quarter; ++i) {
        state = state * 1664525u + 1013904223u;
        pcm[i] = static_cast<int32_t>(static_cast<int32_t>(state >> 11) % 20000);
    }

    for (size_t i = half + quarter; i < total; ++i) {
        pcm[i] = (i % 8 == 0) ? ((i & 1) ? burst : -burst) : 0;
    }

    return pcm;
}

void verify_block_roundtrip(const std::vector<int32_t>& pcm,
                            bool enable_partitioning,
                            uint8_t& out_order,
                            std::vector<uint8_t>& encoded,
                            bool use_zero_run = true) {
    Block::Encoder enc(8);
    enc.set_partitioning_enabled(enable_partitioning);
    enc.set_zero_run_enabled(use_zero_run);
    enc.set_debug_partitions(true);
    enc.set_debug_block_index(0);
    encoded = enc.encode(pcm);
    out_order = read_partition_order(encoded);
    BitReader br(encoded);
    std::vector<int32_t> decoded;
    Block::Decoder dec;
    bool ok = dec.decode(br, static_cast<uint32_t>(pcm.size()), decoded);
    if (!ok) {
        std::cerr << "block decode failed"
                  << " partitioning=" << (enable_partitioning ? 1 : 0)
                  << " order=" << static_cast<int>(out_order)
                  << " encoded_bytes=" << encoded.size()
                  << " first_byte=" << (encoded.empty() ? 0 : static_cast<int>(encoded[0]))
                  << "\n";
    }
    assert(ok);
    if (decoded != pcm) {
        size_t mismatch = 0;
        while (mismatch < pcm.size() && mismatch < decoded.size() && decoded[mismatch] == pcm[mismatch]) {
            ++mismatch;
        }
        std::cerr << "pcm mismatch partitioning=" << (enable_partitioning ? 1 : 0)
                  << " idx=" << mismatch
                  << " expected=" << pcm[mismatch]
                  << " got=" << decoded[mismatch]
                  << " encoded_bytes=" << encoded.size()
                  << "\n";
    }
    assert(decoded == pcm);
}

} // namespace

void run_partitioning_tests() {
#ifdef LAC_ENABLE_PARTITIONING
    if (std::getenv("LAC_RUN_PART_TESTS") == nullptr) {
        std::cout << "Partitioning tests skipped (set LAC_RUN_PART_TESTS=1 to run)\n";
        return;
    }
    {
        std::vector<int32_t> pcm = make_composite_pattern(1024);
        uint8_t order = 0;
        std::vector<uint8_t> encoded;
        verify_block_roundtrip(pcm, true, order, encoded);
        assert(order <= Block::MAX_PARTITION_ORDER);
        std::cout << "[partition-test] order=" << static_cast<int>(order)
                  << " bytes=" << encoded.size() << "\n";
    }

    {
        std::vector<int32_t> pcm = make_composite_pattern(2048);
        uint8_t baseline_order = 0;
        uint8_t new_order = 0;
        std::vector<uint8_t> baseline_encoded;
        std::vector<uint8_t> partition_encoded;
        verify_block_roundtrip(pcm, false, baseline_order, baseline_encoded, false);
        verify_block_roundtrip(pcm, true, new_order, partition_encoded, false);
        assert(partition_encoded.size() <= baseline_encoded.size());
    }

    {
        std::vector<int32_t> left = make_composite_pattern(1536);
        std::vector<int32_t> right(left.rbegin(), left.rend());
        LAC::Encoder enc(12, 0, 44100, 16, false, false, false);
        enc.set_partitioning_enabled(true);
        std::vector<uint8_t> bs = enc.encode(left, right);
        LAC::Decoder dec;
        std::vector<int32_t> outL, outR;
        FrameHeader hdr;
        bool ok = dec.decode(bs.data(), bs.size(), outL, outR, &hdr);
        if (!ok) {
            std::cerr << "lac decode failed bytes=" << bs.size() << "\n";
        }
        assert(ok);
        assert(outL == left);
        assert(outR == right);
        assert(hdr.channels == 2);
    }
#else
    std::cout << "Partitioning disabled at build time; skipping partitioning tests\n";
#endif
}
