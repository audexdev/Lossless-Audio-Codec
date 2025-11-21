#include "decoder.hpp"
#include <cstdlib>
#include <iostream>
#include "codec/bitstream/bit_reader.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"

namespace Block {

Decoder::Decoder() {}

bool Decoder::decode(BitReader& br, uint32_t block_size, std::vector<int32_t>& out) {
    if (block_size == 0 || block_size > MAX_BLOCK_SIZE) return false;
#ifdef LAC_ENABLE_ZERORUN
    const bool debug_zr = (std::getenv("LAC_DEBUG_ZR") != nullptr);
#endif
#ifdef LAC_ENABLE_PARTITIONING
    const bool debug_part = (std::getenv("LAC_DEBUG_PART") != nullptr);
#else
    const bool debug_part = false;
#endif

    auto read_rice_unsigned = [](BitReader& reader, uint32_t k) -> uint32_t {
        uint32_t q = 0;
        while (reader.read_bit() == 1u) {
            ++q;
            if (reader.has_error()) return 0;
        }
        uint32_t remainder = (k > 0) ? reader.read_bits(static_cast<int>(k)) : 0u;
        return (q << k) | remainder;
    };

    auto zigzag_decode = [](uint32_t u) -> int32_t {
        return static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
    };

    auto adapt_k = [](uint64_t sum, uint32_t count, bool stateless) -> uint32_t {
        if (!stateless) return Rice::adapt_k(sum, count);
        if (count == 0) return 0;
        const uint64_t mean = (sum + (count >> 1)) / count;
        uint32_t k = 0;
        while ((1u << k) < mean && k < 31u) {
            ++k;
        }
        return k;
    };

    auto partition_sizes_for_block = [](uint32_t size, uint8_t order) -> std::vector<uint32_t> {
        std::vector<uint32_t> parts;
        if (order == 0) {
            parts.push_back(size);
            return parts;
        }
        const uint32_t base = size >> order;
        if (base == 0) {
            parts.push_back(size);
            return parts;
        }
        const uint32_t count = 1u << order;
        parts.assign(count, base);
        const uint32_t used = base * (count - 1u);
        parts.back() = size - used;
        return parts;
    };

    auto decode_residual_segment = [&](BitReader& reader,
                                       uint32_t samples,
                                       uint32_t initial_k,
                                       uint8_t residual_mode,
                                       std::vector<int32_t>& residual,
                                       size_t offset,
                                       bool stateless) -> bool {
        const bool debug = debug_part;
        if (residual_mode > 1) return false;

        Rice rice;
        uint32_t current_k = initial_k;
        uint64_t sumU = 0;
        uint32_t count = 0;

        if (residual_mode == 0) {
            for (uint32_t i = 0; i < samples; ++i) {
                residual[offset + i] = rice.decode(reader, current_k);
                uint32_t u = static_cast<uint32_t>((residual[offset + i] << 1) ^ (residual[offset + i] >> 31));
                sumU += u;
                ++count;
                current_k = adapt_k(sumU, count, stateless);
                if (reader.has_error()) return false;
            }
            return true;
        }

#ifdef LAC_ENABLE_ZERORUN
        constexpr uint32_t kTagNormal = 0b00u;
        constexpr uint32_t kTagRun = 0b01u;
        constexpr uint32_t kTagEscape = 0b10u;

        uint32_t idx = 0;
        while (idx < samples) {
            uint32_t tag_prefix = reader.read_bit();
            if (reader.has_error()) {
                if (debug) {
                    std::cerr << "[part-dec] read error before tag idx=" << idx << "\n";
                }
                return false;
            }
            uint32_t tag = kTagEscape;
            if (tag_prefix == 0) {
                tag = reader.read_bit();
            } else {
                (void)reader.read_bit(); // consume escape tag padding bit
            }
            if (reader.has_error()) {
                if (debug) {
                    std::cerr << "[part-dec] read error before tag idx=" << idx << "\n";
                }
                return false;
            }
            if (debug) {
                std::cerr << "[part-dec] tag=" << tag << " idx=" << idx << " k=" << current_k << "\n";
            }
            if (tag == kTagNormal) {
                uint32_t u = read_rice_unsigned(reader, current_k);
                if (reader.has_error() || idx >= samples) {
                    if (debug) {
                        std::cerr << "[part-dec] read error in normal idx=" << idx << "\n";
                    }
                    break;
                }
                residual[offset + idx++] = static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
                if (debug && idx <= 8) {
                    std::cerr << "[part-val] idx=" << (offset + idx - 1)
                              << " v=" << residual[offset + idx - 1]
                              << " k=" << current_k << "\n";
                }
                sumU += u;
                ++count;
                current_k = adapt_k(sumU, count, stateless);
#ifdef LAC_ENABLE_ZERORUN
                if (debug_zr) {
                    std::cerr << "[zr-decode] normal idx=" << (offset + idx - 1) << " err=" << reader.has_error() << "\n";
                }
#endif
            } else if (tag == kTagRun) {
                uint32_t run_len = read_rice_unsigned(reader, Block::ZERO_RUN_LENGTH_K) + Block::ZERO_RUN_MIN_LENGTH;
                if (idx + run_len > samples) {
                    if (debug) {
                        std::cerr << "[part-dec] run overflow idx=" << idx
                                  << " run=" << run_len
                                  << " samples=" << samples << "\n";
                    }
                    return false;
                }
                for (uint32_t j = 0; j < run_len; ++j) {
                    residual[offset + idx++] = 0;
                    if (debug && idx <= 8) {
                        std::cerr << "[part-val] idx=" << (offset + idx - 1)
                                  << " v=0 k=" << current_k << "\n";
                    }
                    ++count;
                current_k = adapt_k(sumU, count, stateless);
                }
#ifdef LAC_ENABLE_ZERORUN
                if (debug_zr) {
                    std::cerr << "[zr-decode] run len=" << run_len << " idx=" << idx << " err=" << reader.has_error() << "\n";
                }
#endif
            } else if (tag == kTagEscape) {
                if (idx >= samples) {
                    if (debug) {
                        std::cerr << "[part-dec] escape overflow idx=" << idx
                                  << " samples=" << samples << "\n";
                    }
                    return false;
                }
                uint32_t zz = reader.read_bits(32);
                if (reader.has_error()) {
                    if (debug) {
                        std::cerr << "[part-dec] read error escape idx=" << idx << "\n";
                    }
                    break;
                }
                int32_t value = zigzag_decode(zz);
                residual[offset + idx++] = value;
                if (debug && idx <= 8) {
                    std::cerr << "[part-val] idx=" << (offset + idx - 1)
                              << " v=" << value
                              << " k=" << current_k << "\n";
                }
                uint32_t u = static_cast<uint32_t>((value << 1) ^ (value >> 31));
                sumU += u;
                ++count;
                current_k = adapt_k(sumU, count, stateless);
                if (debug_zr) {
                    std::cerr << "[zr-decode] escape idx=" << idx << " val=" << value << " err=" << reader.has_error() << "\n";
                }
            } else {
                if (debug) {
                    std::cerr << "[part-dec] invalid tag=" << tag
                              << " idx=" << idx
                              << " bits_left=" << reader.bits_remaining() << "\n";
                }
                return false;
            }
        }
        if (idx != samples && debug) {
            std::cerr << "[part-dec] idx_mismatch idx=" << idx << " samples=" << samples << "\n";
        }
        return idx == samples;
#else
        (void)residual;
        (void)offset;
        (void)samples;
        (void)initial_k;
        (void)residual_mode;
        (void)sumU;
        (void)count;
        return false;
#endif
    };

    auto decode_channel_variant = [&](BitReader& reader, bool has_control_byte) -> bool {
        uint8_t partition_order = 0;
        uint8_t residual_mode = 0;

        if (has_control_byte) {
            const uint8_t control = static_cast<uint8_t>(reader.read_bits(8));
            if (reader.has_error()) return false;
            const bool has_partition_flag = (control & Block::PARTITION_FLAG) != 0u;
            if (has_partition_flag) {
                partition_order = static_cast<uint8_t>((control & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT);
                if (partition_order > Block::MAX_PARTITION_ORDER) return false;
                if ((block_size >> partition_order) < Block::MIN_PARTITION_SIZE) return false;
                if (debug_part) {
                    std::cerr << "[part-decode] ctrl=" << static_cast<int>(control)
                              << " p=" << static_cast<int>(partition_order)
                              << " block=" << block_size << "\n";
                }
            } else {
                residual_mode = control;
                if (residual_mode > 1) return false;
#ifndef LAC_ENABLE_ZERORUN
                if (residual_mode == 1) return false;
#endif
                partition_order = 0;
            }
        }

        int order = reader.read_bits(8);
        if (reader.has_error()) return false;
        if (order <= 0 || order > 32 || static_cast<uint32_t>(order) >= block_size) return false;

        uint32_t initial_k = 0;
        if (partition_order == 0) {
            initial_k = reader.read_bits(5);
            if (reader.has_error()) return false;
        }

        std::vector<int16_t> coeffs_q15(order + 1);
        coeffs_q15[0] = 0;

        for (int i = 1; i <= order; ++i) {
            coeffs_q15[i] = int16_t(reader.read_bits(16));
            if (reader.has_error()) return false;
        }

        std::vector<int32_t> residual(block_size);
        const bool stateless = (partition_order > 0);

        if (partition_order == 0) {
            uint8_t mode = has_control_byte ? residual_mode : 0;
            if (!decode_residual_segment(reader, block_size, initial_k, mode, residual, 0, stateless)) {
                return false;
            }
        } else {
            const uint32_t expected_parts = 1u << partition_order;
            const auto part_sizes = partition_sizes_for_block(block_size, partition_order);
            if (part_sizes.size() != expected_parts || part_sizes.back() == 0) {
                return false;
            }
            std::vector<uint8_t> part_modes(expected_parts);
            std::vector<uint32_t> part_k(expected_parts);
            for (uint32_t i = 0; i < expected_parts; ++i) {
                part_modes[i] = static_cast<uint8_t>(reader.read_bit());
                part_k[i] = reader.read_bits(5);
                if (reader.has_error()) return false;
#ifndef LAC_ENABLE_ZERORUN
                if (part_modes[i] == 1) return false;
#endif
            }
            if (debug_part) {
                std::cerr << "[part-decode] modes/k:";
                for (uint32_t i = 0; i < expected_parts; ++i) {
                    std::cerr << " (" << static_cast<int>(part_modes[i]) << "," << part_k[i] << "," << part_sizes[i] << ")";
                }
                std::cerr << " bits_before_parts=" << reader.bits_remaining() << "\n";
            }
            size_t offset = 0;
            for (uint32_t i = 0; i < expected_parts; ++i) {
                size_t bits_before = reader.bits_remaining();
                if (!decode_residual_segment(reader, part_sizes[i], part_k[i], part_modes[i], residual, offset, stateless)) {
                    if (debug_part) {
                        std::cerr << "[part-fail] idx=" << i
                                  << " consumed=" << (bits_before - reader.bits_remaining())
                                  << " size=" << part_sizes[i]
                                  << " k=" << part_k[i]
                                  << " mode=" << static_cast<int>(part_modes[i])
                                  << " bits_left=" << reader.bits_remaining()
                                  << "\n";
                    }
                    return false;
                }
                if (debug_part) {
                    size_t part_end_bit = reader.bits_remaining();
                    std::cerr << "[part-ok] idx=" << i
                              << " bits_consumed=" << (bits_before - part_end_bit)
                              << " bits_left=" << part_end_bit << "\n";
                }
                offset += part_sizes[i];
            }
        }

        reader.align_to_byte();
        if (reader.has_error()) return false;

        LPC lpc(order);

        std::vector<int32_t> pcm;
        lpc.restore_from_residual_q15(residual, coeffs_q15, pcm);
        if (pcm.size() != block_size) return false;
        out.swap(pcm);
        return true;
    };

    BitReader with_control = br;
    if (decode_channel_variant(with_control, true)) {
        br = with_control;
        return true;
    }

    BitReader legacy = br;
    if (decode_channel_variant(legacy, false)) {
        br = legacy;
        return true;
    }
    return false;
}

} // namespace Block
