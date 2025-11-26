#include "codec/block/constants.hpp"
#include "codec/block/decoder.hpp"
#include "codec/block/encoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/bitstream/bit_writer.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/lac/decoder.hpp"
#include "codec/lac/encoder.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

uint8_t read_partition_order(const std::vector<uint8_t>& buffer) {
    if (buffer.empty()) return 0;
    BitReader br(buffer);
    uint8_t predictor_type = static_cast<uint8_t>(br.read_bits(8));
    uint32_t order = br.read_bits(8);
    if (predictor_type == 2) {
        for (uint32_t i = 0; i < order; ++i) {
            (void)br.read_bits(16);
        }
    }
    uint32_t control = br.read_bits(8);
    if ((control & Block::PARTITION_FLAG) == 0u) return 0;
    return static_cast<uint8_t>((control & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT);
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

std::vector<uint8_t> make_manual_partition_block(uint8_t order) {
    const uint32_t block_size = Block::MIN_PARTITION_SIZE << order;
    BitWriter bw;
    bw.write_bits(static_cast<uint32_t>(2), 8); // predictor_type = LPC for compatibility
    const uint8_t lpc_order = 4;
    bw.write_bits(static_cast<uint32_t>(lpc_order), 8);
    for (int i = 0; i < lpc_order; ++i) {
        bw.write_bits(0u, 16);
    }
    uint8_t control = static_cast<uint8_t>(0u << 5);
    if (order > 0) {
        control |= Block::PARTITION_FLAG;
        control |= static_cast<uint8_t>((order & Block::PARTITION_ORDER_MASK) << Block::PARTITION_ORDER_SHIFT);
    }
    bw.write_bits(control, 8);
    const uint32_t parts = 1u << order;
    for (uint32_t i = 0; i < parts; ++i) {
        bw.write_bits(0u, 2); // mode = Rice
        bw.write_bits(0u, 5); // k = 0
    }

    const uint32_t part_size = block_size >> order;
    for (uint32_t p = 0; p < parts; ++p) {
        for (uint32_t i = 0; i < part_size; ++i) {
            bw.write_bit(0u); // Rice code for zero with k=0
        }
    }
    bw.flush_to_byte();
    return bw.get_buffer();
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
    {
        // Manual buffers for all supported partition orders.
        for (uint8_t order = 0; order <= Block::MAX_PARTITION_ORDER; ++order) {
            std::vector<uint8_t> buf = make_manual_partition_block(order);
            BitReader br(buf);
            Block::Decoder dec;
            std::vector<int32_t> decoded;
            uint32_t block_size = Block::MIN_PARTITION_SIZE << order;
            assert(dec.decode(br, block_size, decoded));
            assert(decoded.size() == block_size);
            for (int32_t v : decoded) {
                assert(v == 0);
            }
        }
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
        // End-to-end LAC path omitted here; covered elsewhere.
    }
}
