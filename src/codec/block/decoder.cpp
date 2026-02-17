#include "decoder.hpp"
#include <iostream>
#include "codec/bitstream/bit_reader.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "utils/logger.hpp"
#include <optional>
#include <sstream>
#include <stdexcept>

namespace Block {

Decoder::Decoder() {}

bool Decoder::decode(BitReader& br, uint32_t block_size, std::vector<int32_t>& out) {
    if (block_size == 0 || block_size > MAX_BLOCK_SIZE) return false;
    const bool debug_zr = LAC_DEBUG_ZR_ENABLED();
    const bool debug_part = LAC_DEBUG_PART_ENABLED();
    constexpr uint32_t kBinTagZero = 0b00u;
    constexpr uint32_t kBinTagOne = 0b01u;
    constexpr uint32_t kBinTagTwo = 0b10u;
    constexpr uint32_t kBinTagFallback = 0b11u;

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

    auto adapt_k = [](uint64_t sum, uint32_t count, bool stateless, Rice::AdaptState* state) -> uint32_t {
        if (!stateless) return Rice::adapt_k(sum, count, *state);
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
        if (residual_mode > 2) return false;

        Rice rice;
        uint32_t current_k = initial_k;
        uint64_t sumU = 0;
        uint32_t count = 0;
        std::optional<Rice::AdaptState> adapt_state_opt;
        if (!stateless) adapt_state_opt.emplace();
        Rice::AdaptState* adapt_state = adapt_state_opt ? &*adapt_state_opt : nullptr;
        auto to_unsigned = [](int32_t v) -> uint32_t {
            return (static_cast<uint32_t>(v) << 1) ^ (static_cast<uint32_t>(v >> 31));
        };

        if (residual_mode == 0) {
            for (uint32_t i = 0; i < samples; ++i) {
                residual[offset + i] = rice.decode(reader, current_k);
                uint32_t u = (static_cast<uint32_t>(residual[offset + i]) << 1)
                    ^ (static_cast<uint32_t>(residual[offset + i] >> 31));
                sumU += u;
                ++count;
                current_k = adapt_k(sumU, count, stateless, adapt_state);
                if (reader.has_error()) return false;
            }
            return true;
        }

        if (residual_mode == 1) {
            constexpr uint32_t kTagNormal = 0b00u;
            constexpr uint32_t kTagRun = 0b01u;
            constexpr uint32_t kTagEscape = 0b10u;

            uint32_t idx = 0;
            while (idx < samples) {
                uint32_t tag_prefix = reader.read_bit();
                if (reader.has_error()) {
                    if (debug) {
                        LAC_DEBUG_LOG("[part-dec] read error before tag idx=" << idx << "\n");
                    }
                    return false;
                }
                uint32_t tag_suffix = reader.read_bit();
                if (reader.has_error()) {
                    if (debug) {
                        LAC_DEBUG_LOG("[part-dec] read error before tag idx=" << idx << "\n");
                    }
                    return false;
                }
                uint32_t tag = 0xFFu;
                if (tag_prefix == 0u) {
                    tag = (tag_suffix == 0u) ? kTagNormal : kTagRun; // 00 / 01
                } else if (tag_suffix == 0u) {
                    tag = kTagEscape; // 10
                } else {
                    if (debug) {
                        LAC_DEBUG_LOG("[part-dec] invalid tag=3 idx=" << idx << "\n");
                    }
                    return false; // 11 is reserved/invalid
                }
                if (debug) {
                    LAC_DEBUG_LOG("[part-dec] tag=" << tag << " idx=" << idx << " k=" << current_k << "\n");
                }
                if (tag == kTagNormal) {
                    uint32_t u = read_rice_unsigned(reader, current_k);
                    if (reader.has_error() || idx >= samples) {
                        if (debug) {
                            LAC_DEBUG_LOG("[part-dec] read error in normal idx=" << idx << "\n");
                        }
                        break;
                    }
                    residual[offset + idx++] = static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
                    if (debug && idx <= 8) {
                        LAC_DEBUG_LOG("[part-val] idx=" << (offset + idx - 1)
                                      << " v=" << residual[offset + idx - 1]
                                      << " k=" << current_k << "\n");
                    }
                    sumU += u;
                    ++count;
                    current_k = adapt_k(sumU, count, stateless, adapt_state);
                    if (debug_zr) {
                        LAC_DEBUG_LOG("[zr-decode] normal idx=" << (offset + idx - 1) << " err=" << reader.has_error() << "\n");
                    }
                } else if (tag == kTagRun) {
                    uint32_t run_len = read_rice_unsigned(reader, Block::ZERO_RUN_LENGTH_K) + Block::ZERO_RUN_MIN_LENGTH;
                    if (idx + run_len > samples) {
                        if (debug) {
                            LAC_DEBUG_LOG("[part-dec] run overflow idx=" << idx
                                          << " run=" << run_len
                                          << " samples=" << samples << "\n");
                        }
                        return false;
                    }
                    for (uint32_t j = 0; j < run_len; ++j) {
                        residual[offset + idx++] = 0;
                        if (debug && idx <= 8) {
                            LAC_DEBUG_LOG("[part-val] idx=" << (offset + idx - 1)
                                          << " v=0 k=" << current_k << "\n");
                        }
                        ++count;
                        current_k = adapt_k(sumU, count, stateless, adapt_state);
                    }
                    if (debug_zr) {
                        LAC_DEBUG_LOG("[zr-decode] run len=" << run_len << " idx=" << idx << " err=" << reader.has_error() << "\n");
                    }
                } else if (tag == kTagEscape) {
                    if (idx >= samples) {
                        if (debug) {
                            LAC_DEBUG_LOG("[part-dec] escape overflow idx=" << idx
                                          << " samples=" << samples << "\n");
                        }
                        return false;
                    }
                    uint32_t zz = reader.read_bits(32);
                    if (reader.has_error()) {
                        if (debug) {
                            LAC_DEBUG_LOG("[part-dec] read error escape idx=" << idx << "\n");
                        }
                        break;
                    }
                    int32_t value = zigzag_decode(zz);
                    residual[offset + idx++] = value;
                    if (debug && idx <= 8) {
                        LAC_DEBUG_LOG("[part-val] idx=" << (offset + idx - 1)
                                      << " v=" << value
                                      << " k=" << current_k << "\n");
                    }
                    uint32_t u = (static_cast<uint32_t>(value) << 1)
                        ^ (static_cast<uint32_t>(value >> 31));
                    sumU += u;
                    ++count;
                    current_k = adapt_k(sumU, count, stateless, adapt_state);
                    if (debug_zr) {
                        LAC_DEBUG_LOG("[zr-decode] escape idx=" << idx << " val=" << value << " err=" << reader.has_error() << "\n");
                    }
                } else {
                    if (debug) {
                        LAC_DEBUG_LOG("[part-dec] invalid tag=" << tag
                                      << " idx=" << idx
                                      << " bits_left=" << reader.bits_remaining() << "\n");
                    }
                    return false;
                }
            }
            if (idx != samples && debug) {
                LAC_DEBUG_LOG("[part-dec] idx_mismatch idx=" << idx << " samples=" << samples << "\n");
            }
            return idx == samples;
        }

        if (residual_mode == 2) {
            uint32_t idx = 0;
            while (idx < samples) {
                uint32_t tag = reader.read_bits(2);
                if (reader.has_error()) return false;
                int32_t value = 0;
                if (tag == kBinTagZero) {
                    value = 0;
                } else if (tag == kBinTagOne) {
                    uint32_t sign = reader.read_bit();
                    if (reader.has_error()) return false;
                    value = (sign == 0) ? 1 : -1;
                } else if (tag == kBinTagTwo) {
                    uint32_t sign = reader.read_bit();
                    if (reader.has_error()) return false;
                    value = (sign == 0) ? 2 : -2;
                } else if (tag == kBinTagFallback) {
                    uint32_t u = read_rice_unsigned(reader, current_k);
                    if (reader.has_error()) return false;
                    value = zigzag_decode(u);
                    sumU += u;
                    ++count;
                    current_k = adapt_k(sumU, count, stateless, adapt_state);
                    residual[offset + idx++] = value;
                    continue;
                } else {
                    return false;
                }
                uint32_t u = to_unsigned(value);
                residual[offset + idx++] = value;
                sumU += u;
                ++count;
                current_k = adapt_k(sumU, count, stateless, adapt_state);
            }
            return idx == samples;
        }

        return false;
    };

    auto restore_fixed = [&](const std::vector<int32_t>& residual, int order, std::vector<int32_t>& pcm) {
        pcm.resize(residual.size(), 0);
        if (order == 0) {
            pcm = residual;
            return;
        }
        for (int i = 0; i < order && i < static_cast<int>(residual.size()); ++i) {
            pcm[i] = residual[i];
        }
        for (size_t i = static_cast<size_t>(order); i < residual.size(); ++i) {
            int64_t pred = 0;
            switch (order) {
                case 1:
                    pred = pcm[i - 1];
                    break;
                case 2:
                    pred = 2LL * pcm[i - 1] - pcm[i - 2];
                    break;
                case 3:
                    pred = 3LL * pcm[i - 1] - 3LL * pcm[i - 2] + pcm[i - 3];
                    break;
                case 4:
                    pred = 4LL * pcm[i - 1] - 6LL * pcm[i - 2] + 4LL * pcm[i - 3] - pcm[i - 4];
                    break;
                default:
                    pred = 0;
                    break;
            }
            pcm[i] = static_cast<int32_t>(residual[i] + pred);
        }
    };

    auto restore_fir = [&](const std::vector<int32_t>& residual, int taps, std::vector<int32_t>& pcm) {
        pcm.resize(residual.size(), 0);
        for (int i = 0; i < taps && i < static_cast<int>(residual.size()); ++i) {
            pcm[i] = residual[i];
        }
        for (size_t i = static_cast<size_t>(taps); i < residual.size(); ++i) {
            int64_t pred = 0;
            pred += 3LL * pcm[i - 1];
            pred += -1LL * pcm[i - 2];
            pred >>= 2;
            pcm[i] = static_cast<int32_t>(residual[i] + pred);
        }
    };

    // Header layout: predictor_type (8), predictor_order (8), LPC coeffs (if LPC),
    // residual_control (8), partition metadata ([mode:2|k:5] * partitions), then residuals.
    uint8_t predictor_type = static_cast<uint8_t>(br.read_bits(8));
    int order = static_cast<int>(br.read_bits(8));
    if (br.has_error()) return false;
    if (predictor_type > 2) return false;
    if (predictor_type == 2) {
        if (order <= 0 || order > 32 || static_cast<uint32_t>(order) >= block_size) return false;
    } else if (predictor_type == 1) {
        if (order < 2 || order > 32) {
            throw std::runtime_error("[block-decode] invalid FIR predictor order");
        }
    } else {
        if (order < 0 || order > 32) return false;
    }

    std::vector<int16_t> coeffs_q15;
    if (predictor_type == 2) {
        coeffs_q15.assign(order + 1, 0);
        for (int i = 1; i <= order; ++i) {
            coeffs_q15[i] = int16_t(br.read_bits(16));
            if (br.has_error()) return false;
        }
    }

    const uint8_t control = static_cast<uint8_t>(br.read_bits(8));
    if (br.has_error()) return false;
    const bool partition_flag = (control & Block::PARTITION_FLAG) != 0u;
    uint8_t partition_order = partition_flag
        ? static_cast<uint8_t>((control & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT)
        : 0;
    const uint8_t control_mode = static_cast<uint8_t>((control >> 5) & 0x03u);
    if (partition_flag && partition_order == 0) return false;
    if (partition_order > Block::MAX_PARTITION_ORDER) return false;
    if ((partition_order > 0) && ((block_size >> partition_order) < Block::MIN_PARTITION_SIZE)) return false;

    const uint32_t partition_count = (partition_order == 0) ? 1u : (1u << partition_order);
    const auto part_sizes = partition_sizes_for_block(block_size, partition_order);
    if (part_sizes.size() != partition_count || part_sizes.back() == 0) return false;

    std::vector<uint8_t> part_modes(partition_count);
    std::vector<uint32_t> part_k(partition_count);
    for (uint32_t i = 0; i < partition_count; ++i) {
        part_modes[i] = static_cast<uint8_t>(br.read_bits(2));
        part_k[i] = br.read_bits(5);
        if (br.has_error()) return false;
        if (part_modes[i] > 2) return false;
    }

    if (debug_part) {
        LAC_DEBUG_LOG("[block-hdr] predict=" << static_cast<int>(predictor_type)
                      << " order=" << order
                      << " ctrl=" << static_cast<int>(control)
                      << " mode=" << static_cast<int>(control_mode)
                      << " pflag=" << (partition_flag ? 1 : 0)
                      << " porder=" << static_cast<int>(partition_order)
                      << " parts=" << partition_count
                      << " block=" << block_size
                      << "\n");
        std::ostringstream oss;
        oss << "[part-decode] modes/k:";
        for (uint32_t i = 0; i < partition_count; ++i) {
            oss << " (" << static_cast<int>(part_modes[i]) << "," << part_k[i] << "," << part_sizes[i] << ")";
        }
        oss << " bits_before_parts=" << br.bits_remaining() << "\n";
        LAC_DEBUG_LOG(oss.str());
    }

    if (partition_order == 0 && part_modes[0] != control_mode) {
        if (debug_part) {
            LAC_DEBUG_LOG("[block-hdr] mode-mismatch control=" << static_cast<int>(control_mode)
                          << " meta=" << static_cast<int>(part_modes[0]) << "\n");
        }
    }

    std::vector<int32_t> residual(block_size);
    const bool stateless = (partition_order > 0);
    size_t offset = 0;
    for (uint32_t i = 0; i < partition_count; ++i) {
        size_t bits_before = br.bits_remaining();
        if (debug_part) {
            LAC_DEBUG_LOG("[part-entry] idx=" << i
                          << " mode=" << static_cast<int>(part_modes[i])
                          << " k=" << part_k[i]
                          << " samples=" << part_sizes[i]
                          << " bits_before=" << bits_before
                          << "\n");
        }
        if (!decode_residual_segment(br, part_sizes[i], part_k[i], part_modes[i], residual, offset, stateless)) {
            if (debug_part) {
                LAC_DEBUG_LOG("[part-fail] idx=" << i
                              << " consumed=" << (bits_before - br.bits_remaining())
                              << " size=" << part_sizes[i]
                              << " k=" << part_k[i]
                              << " mode=" << static_cast<int>(part_modes[i])
                              << " bits_left=" << br.bits_remaining()
                              << "\n");
            }
            return false;
        }
        if (debug_part) {
            size_t part_end_bit = br.bits_remaining();
            LAC_DEBUG_LOG("[part-ok] idx=" << i
                          << " bits_consumed=" << (bits_before - part_end_bit)
                          << " bits_left=" << part_end_bit << "\n");
        }
        offset += part_sizes[i];
    }
    if (offset != block_size) return false;

    br.align_to_byte();
    if (br.has_error()) return false;

    std::vector<int32_t> pcm;
    if (predictor_type == 0) {
        restore_fixed(residual, order, pcm);
    } else if (predictor_type == 1) {
        restore_fir(residual, order, pcm);
    } else {
        LPC lpc(order);
        lpc.restore_from_residual_q15(residual, coeffs_q15, pcm);
    }
    if (pcm.size() != block_size) return false;
    out.swap(pcm);
    return true;
}

} // namespace Block
