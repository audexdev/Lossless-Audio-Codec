#include "decoder.hpp"
#include <iostream>
#include "codec/bitstream/bit_reader.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

namespace Block {

Decoder::Decoder() {}

template <size_t... I>
int64_t lpc_accumulate_known_order(const int32_t* pcm,
                                   size_t n,
                                   const std::array<int16_t, 33>& coeffs,
                                   std::index_sequence<I...>) {
    int64_t acc = 0;
    ((acc += static_cast<int64_t>(coeffs[I + 1]) *
             static_cast<int64_t>(pcm[n - (I + 1)])), ...);
    return acc;
}

template <int Order>
bool restore_lpc_known_order_in_place(int32_t* pcm,
                                      size_t size,
                                      const std::array<int16_t, 33>& coeffs,
                                      int64_t lo,
                                      int64_t hi) {
    const size_t warmup = std::min(size, static_cast<size_t>(Order));
    for (size_t n = 0; n < warmup; ++n) {
        int64_t acc = 0;
        for (int i = 1; i <= static_cast<int>(n); ++i) {
            acc += static_cast<int64_t>(coeffs[i]) *
                   static_cast<int64_t>(pcm[n - static_cast<size_t>(i)]);
        }
        const int64_t sample = (acc >> 15) + static_cast<int64_t>(pcm[n]);
        if (sample < lo || sample > hi) return false;
        pcm[n] = static_cast<int32_t>(sample);
    }
    for (size_t n = warmup; n < size; ++n) {
        const int64_t acc =
            lpc_accumulate_known_order(pcm, n, coeffs, std::make_index_sequence<Order>{});
        const int64_t sample = (acc >> 15) + static_cast<int64_t>(pcm[n]);
        if (sample < lo || sample > hi) return false;
        pcm[n] = static_cast<int32_t>(sample);
    }
    return true;
}

bool Decoder::decode(BitReader& br, uint32_t block_size, std::vector<int32_t>& out) {
    std::vector<int32_t> pcm(block_size);
    if (!this->decode_into(br, block_size, pcm.data())) return false;
    out.swap(pcm);
    return true;
}

bool Decoder::decode_into(BitReader& br, uint32_t block_size, int32_t* out) {
    if (block_size == 0 || block_size > MAX_BLOCK_SIZE || out == nullptr) return false;
    const bool debug_zr = LAC_DEBUG_ZR_ENABLED();
    const bool debug_part = LAC_DEBUG_PART_ENABLED();
    constexpr uint32_t kBinTagZero = 0b00u;
    constexpr uint32_t kBinTagOne = 0b01u;
    constexpr uint32_t kBinTagTwo = 0b10u;
    constexpr uint32_t kBinTagFallback = 0b11u;
    constexpr uint8_t kResidualModeStaticRice = 3u;

    auto read_rice_unsigned = [](BitReader& reader, uint32_t k, uint32_t& value) -> bool {
        if (k > 31u) return false;
        uint32_t q = 0;
        const uint32_t max_q = std::numeric_limits<uint32_t>::max() >> k;
        if (!reader.read_unary_ones(max_q, q)) return false;
        uint32_t remainder = (k > 0) ? reader.read_bits(static_cast<int>(k)) : 0u;
        if (reader.has_error()) return false;
        value = (q << k) | remainder;
        return true;
    };

    auto zigzag_decode = [](uint32_t u) -> int32_t {
        if ((u & 1u) == 0u) return static_cast<int32_t>(u >> 1);
        return static_cast<int32_t>(-static_cast<int64_t>((u >> 1) + 1u));
    };

    auto adapt_k = [](uint64_t sum, uint32_t count, bool stateless, Rice::AdaptState* state) -> uint32_t {
        if (!stateless) return Rice::adapt_k(sum, count, *state);
        if (count == 0) return 0;
        const uint64_t mean = (sum + (count >> 1)) / count;
        if (mean <= 1) return 0;
        return std::min<uint32_t>(31u, std::bit_width(mean - uint64_t{1}));
    };

    auto partition_size_at = [](uint32_t size, uint8_t order, uint32_t index, uint32_t count) -> uint32_t {
        if (order == 0) return size;
        const uint32_t base = size >> order;
        return (index + 1u == count) ? (size - base * (count - 1u)) : base;
    };

    auto decode_residual_segment = [&](BitReader& reader,
                                       uint32_t samples,
                                       uint32_t initial_k,
                                       uint8_t residual_mode,
                                       int32_t* residual,
                                       size_t offset,
                                       bool stateless) -> bool {
        const bool debug = debug_part;
        if (residual_mode > kResidualModeStaticRice) return false;

        uint32_t current_k = initial_k;
        uint64_t sumU = 0;
        uint32_t count = 0;
        std::optional<Rice::AdaptState> adapt_state_opt;
        if (!stateless) adapt_state_opt.emplace();
        Rice::AdaptState* adapt_state = adapt_state_opt ? &*adapt_state_opt : nullptr;
        auto to_unsigned = [](int32_t v) -> uint32_t {
            const uint32_t sign_mask =
                (v < 0) ? std::numeric_limits<uint32_t>::max() : 0u;
            return (static_cast<uint32_t>(v) << 1) ^ sign_mask;
        };

        if (residual_mode == 0) {
            for (uint32_t i = 0; i < samples; ++i) {
                uint32_t u = 0;
                if (!read_rice_unsigned(reader, current_k, u)) return false;
                residual[offset + i] = zigzag_decode(u);
                sumU += u;
                ++count;
                current_k = adapt_k(sumU, count, stateless, adapt_state);
            }
            return true;
        }

        if (residual_mode == 1) {
            constexpr uint32_t kTagNormal = 0b00u;
            constexpr uint32_t kTagRun = 0b01u;
            constexpr uint32_t kTagEscape = 0b10u;

            uint32_t idx = 0;
            while (idx < samples) {
                uint32_t tag = reader.read_bits(2);
                if (reader.has_error()) {
                    if (debug) {
                        LAC_DEBUG_LOG("[part-dec] read error before tag idx=" << idx << "\n");
                    }
                    return false;
                }
                if (tag > kTagEscape) {
                    if (debug) {
                        LAC_DEBUG_LOG("[part-dec] invalid tag=3 idx=" << idx << "\n");
                    }
                    return false; // 11 is reserved/invalid
                }
                if (debug) {
                    LAC_DEBUG_LOG("[part-dec] tag=" << tag << " idx=" << idx << " k=" << current_k << "\n");
                }
                if (tag == kTagNormal) {
                    uint32_t u = 0;
                    if (!read_rice_unsigned(reader, current_k, u) || idx >= samples) {
                        if (debug) {
                            LAC_DEBUG_LOG("[part-dec] read error in normal idx=" << idx << "\n");
                        }
                        break;
                    }
                    residual[offset + idx++] = zigzag_decode(u);
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
                    uint32_t encoded_run_len = 0;
                    if (!read_rice_unsigned(reader, Block::ZERO_RUN_LENGTH_K, encoded_run_len) ||
                        encoded_run_len > std::numeric_limits<uint32_t>::max() - Block::ZERO_RUN_MIN_LENGTH) {
                        return false;
                    }
                    uint32_t run_len = encoded_run_len + Block::ZERO_RUN_MIN_LENGTH;
                    if (run_len > samples - idx) {
                        if (debug) {
                            LAC_DEBUG_LOG("[part-dec] run overflow idx=" << idx
                                          << " run=" << run_len
                                          << " samples=" << samples << "\n");
                        }
                        return false;
                    }
                    std::fill_n(residual + offset + idx, run_len, 0);
                    if (debug && idx < 8) {
                        LAC_DEBUG_LOG("[part-val] idx=" << (offset + idx)
                                      << " v=0 k=" << current_k << "\n");
                    }
                    idx += run_len;
                    if (!stateless) {
                        for (uint32_t j = 0; j < run_len; ++j) {
                            ++count;
                            current_k = adapt_k(sumU, count, false, adapt_state);
                        }
                    }
                    if (stateless) {
                        count += run_len;
                        current_k = adapt_k(sumU, count, true, nullptr);
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
                    uint32_t u = to_unsigned(value);
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
                    uint32_t u = 0;
                    if (!read_rice_unsigned(reader, current_k, u)) return false;
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

        if (residual_mode == kResidualModeStaticRice) {
            for (uint32_t i = 0; i < samples; ++i) {
                uint32_t u = 0;
                if (!read_rice_unsigned(reader, initial_k, u)) return false;
                residual[offset + i] = zigzag_decode(u);
            }
            return true;
        }

        return false;
    };

    auto restore_fixed_in_place = [&](int32_t* pcm, size_t size, int order) -> bool {
        auto restore_with_prediction = [&](auto predict, size_t start) -> bool {
            for (size_t i = start; i < size; ++i) {
                const int64_t sample = static_cast<int64_t>(pcm[i]) + predict(i);
                if (sample < std::numeric_limits<int32_t>::min() ||
                    sample > std::numeric_limits<int32_t>::max()) {
                    return false;
                }
                pcm[i] = static_cast<int32_t>(sample);
            }
            return true;
        };
        switch (order) {
            case 0:
                return true;
            case 1:
                return restore_with_prediction([&](size_t i) {
                    return static_cast<int64_t>(pcm[i - 1]);
                }, 1);
            case 2:
                return restore_with_prediction([&](size_t i) {
                    return 2LL * pcm[i - 1] - pcm[i - 2];
                }, 2);
            case 3:
                return restore_with_prediction([&](size_t i) {
                    return 3LL * pcm[i - 1] - 3LL * pcm[i - 2] + pcm[i - 3];
                }, 3);
            case 4:
                return restore_with_prediction([&](size_t i) {
                    return 4LL * pcm[i - 1] - 6LL * pcm[i - 2] + 4LL * pcm[i - 3] - pcm[i - 4];
                }, 4);
            default:
                return false;
        }
    };

    auto restore_fir_in_place = [&](int32_t* pcm, size_t size, int taps) -> bool {
        for (size_t i = static_cast<size_t>(taps); i < size; ++i) {
            int64_t pred = 0;
            pred += 3LL * pcm[i - 1];
            pred += -1LL * pcm[i - 2];
            pred >>= 2;
            const int64_t sample = static_cast<int64_t>(pcm[i]) + pred;
            if (sample < std::numeric_limits<int32_t>::min() ||
                sample > std::numeric_limits<int32_t>::max()) {
                return false;
            }
            pcm[i] = static_cast<int32_t>(sample);
        }
        return true;
    };

    auto restore_lpc_in_place = [&](int32_t* pcm,
                                    size_t size,
                                    int lpc_order,
                                    const std::array<int16_t, 33>& coeffs) -> bool {
        const int coeff_order = std::max(0, lpc_order);
        const int64_t lo = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
        const int64_t hi = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        switch (coeff_order) {
            case 4:
                return restore_lpc_known_order_in_place<4>(pcm, size, coeffs, lo, hi);
            case 6:
                return restore_lpc_known_order_in_place<6>(pcm, size, coeffs, lo, hi);
            case 8:
                return restore_lpc_known_order_in_place<8>(pcm, size, coeffs, lo, hi);
            case 10:
                return restore_lpc_known_order_in_place<10>(pcm, size, coeffs, lo, hi);
            case 12:
                return restore_lpc_known_order_in_place<12>(pcm, size, coeffs, lo, hi);
            default:
                break;
        }
        const size_t warmup = std::min(size, static_cast<size_t>(coeff_order));
        for (size_t n = 0; n < warmup; ++n) {
            int64_t acc = 0;
            for (int i = 1; i <= static_cast<int>(n); ++i) {
                acc += static_cast<int64_t>(coeffs[i]) *
                       static_cast<int64_t>(pcm[n - i]);
            }
            const int64_t sample = (acc >> 15) + static_cast<int64_t>(pcm[n]);
            if (sample < lo || sample > hi) return false;
            pcm[n] = static_cast<int32_t>(sample);
        }
        for (size_t n = warmup; n < size; ++n) {
            int64_t acc = 0;
            for (int i = 1; i <= coeff_order; ++i) {
                acc += static_cast<int64_t>(coeffs[i]) *
                       static_cast<int64_t>(pcm[n - static_cast<size_t>(i)]);
            }
            const int64_t sample = (acc >> 15) + static_cast<int64_t>(pcm[n]);
            if (sample < lo || sample > hi) return false;
            pcm[n] = static_cast<int32_t>(sample);
        }
        return true;
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
        if (order != 2) return false;
    } else {
        if (order < 0 || order > 4) return false;
    }

    std::array<int16_t, 33> coeffs_q15{};
    if (predictor_type == 2) {
        for (int i = 1; i <= order; ++i) {
            coeffs_q15[i] = int16_t(br.read_bits(16));
            if (br.has_error()) return false;
        }
    }

    const uint8_t control = static_cast<uint8_t>(br.read_bits(8));
    if (br.has_error()) return false;
    if ((control & Block::RESIDUAL_RESERVED_MASK) != 0u) return false;
    const bool partition_flag = (control & Block::PARTITION_FLAG) != 0u;
    const uint8_t partition_order =
        static_cast<uint8_t>((control & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT);
    const uint8_t control_mode = static_cast<uint8_t>((control >> 5) & 0x03u);
    if (control_mode > kResidualModeStaticRice) return false;
    if (partition_flag && partition_order == 0) return false;
    if (!partition_flag && partition_order != 0) return false;
    if (partition_order > Block::MAX_PARTITION_ORDER) return false;
    if ((partition_order > 0) && ((block_size >> partition_order) < Block::MIN_PARTITION_SIZE)) return false;

    const uint32_t partition_count = (partition_order == 0) ? 1u : (1u << partition_order);
    constexpr size_t kMaxPartitionCount = size_t{1} << Block::MAX_PARTITION_ORDER;
    if (partition_count > kMaxPartitionCount ||
        partition_size_at(block_size, partition_order, partition_count - 1u, partition_count) == 0) {
        return false;
    }

    std::array<uint8_t, kMaxPartitionCount> part_modes{};
    std::array<uint32_t, kMaxPartitionCount> part_k{};
    for (uint32_t i = 0; i < partition_count; ++i) {
        part_modes[i] = static_cast<uint8_t>(br.read_bits(2));
        part_k[i] = br.read_bits(5);
        if (br.has_error()) return false;
        if (part_modes[i] > kResidualModeStaticRice) return false;
    }
    if (part_modes[0] != control_mode) return false;

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
            const uint32_t len = partition_size_at(block_size, partition_order, i, partition_count);
            oss << " (" << static_cast<int>(part_modes[i]) << "," << part_k[i] << "," << len << ")";
        }
        oss << " bits_before_parts=" << br.bits_remaining() << "\n";
        LAC_DEBUG_LOG(oss.str());
    }

    const bool stateless = (partition_order > 0);
    size_t offset = 0;
    for (uint32_t i = 0; i < partition_count; ++i) {
        const uint32_t part_size = partition_size_at(block_size, partition_order, i, partition_count);
        size_t bits_before = br.bits_remaining();
        if (debug_part) {
            LAC_DEBUG_LOG("[part-entry] idx=" << i
                          << " mode=" << static_cast<int>(part_modes[i])
                          << " k=" << part_k[i]
                          << " samples=" << part_size
                          << " bits_before=" << bits_before
                          << "\n");
        }
        if (!decode_residual_segment(br, part_size, part_k[i], part_modes[i], out, offset, stateless)) {
            if (debug_part) {
                LAC_DEBUG_LOG("[part-fail] idx=" << i
                              << " consumed=" << (bits_before - br.bits_remaining())
                              << " size=" << part_size
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
        offset += part_size;
    }
    if (offset != block_size) return false;

    if (!br.consume_zero_padding_to_byte()) return false;

    if (predictor_type == 0) {
        return restore_fixed_in_place(out, block_size, order);
    } else if (predictor_type == 1) {
        return restore_fir_in_place(out, block_size, order);
    }
    return restore_lpc_in_place(out, block_size, order, coeffs_q15);
}

} // namespace Block
