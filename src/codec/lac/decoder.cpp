#include "decoder.hpp"
#include <algorithm>
#include <future>
#include <thread>
#include <cassert>
#include <iostream>
#include <sstream>
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "utils/logger.hpp"

namespace LAC {
namespace {

constexpr uint32_t MAX_BLOCK_COUNT = 10'000'000;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz
constexpr uint32_t kBinTagZero = 0b00u;
constexpr uint32_t kBinTagOne = 0b01u;
constexpr uint32_t kBinTagTwo = 0b10u;
constexpr uint32_t kBinTagFallback = 0b11u;
constexpr uint8_t kPredictorFixed = 0;
constexpr uint8_t kPredictorFir = 1;
constexpr uint8_t kPredictorLpc = 2;

bool e2e_debug_enabled() {
    static bool enabled = (std::getenv("LAC_E2E_DEBUG") != nullptr);
    return enabled;
}

class BitSliceReader {
public:
    BitSliceReader(const uint8_t* data, size_t start_bit, size_t bit_length)
        : data(data), bit_pos(start_bit), bit_start(start_bit), bit_end(start_bit + bit_length), error(false) {}

    uint32_t read_bit() {
        if (this->bit_pos >= this->bit_end) {
            this->error = true;
            return 0;
        }
        size_t byte_index = this->bit_pos >> 3;
        int shift = 7 - static_cast<int>(this->bit_pos & 7);
        uint32_t bit = (static_cast<uint32_t>(this->data[byte_index]) >> shift) & 1u;
        ++this->bit_pos;
        return bit;
    }

    uint32_t read_bits(int nbits) {
        uint32_t value = 0;
        for (int i = 0; i < nbits; ++i) {
            value = (value << 1) | this->read_bit();
            if (this->error) break;
        }
        return value;
    }

    void align_to_byte() {
        size_t mod = this->bit_pos & 7u;
        if (mod != 0) {
            this->bit_pos += (8u - mod);
            if (this->bit_pos > this->bit_end) {
                this->bit_pos = this->bit_end;
                this->error = true;
            }
        }
    }

    bool has_error() const {
        return this->error || this->bit_pos > this->bit_end;
    }

    size_t bit_position() const {
        return this->bit_pos;
    }

    size_t bits_remaining() const {
        return (this->bit_end > this->bit_pos) ? (this->bit_end - this->bit_pos) : 0;
    }

    void copy_from(const BitSliceReader& other) {
        this->data = other.data;
        this->bit_pos = other.bit_pos;
        this->bit_start = other.bit_start;
        this->bit_end = other.bit_end;
        this->error = other.error;
    }

    void set_position(size_t pos) {
        this->bit_pos = pos;
        if (this->bit_pos > this->bit_end) {
            this->bit_pos = this->bit_end;
            this->error = true;
        }
    }

private:
    const uint8_t* data;
    size_t bit_pos;
    size_t bit_start;
    size_t bit_end;
    bool error;
};

struct PartitionSlice {
    size_t start_bit = 0;
    size_t bit_length = 0;
    uint32_t sample_count = 0;
    uint8_t residual_mode = 0;
    uint32_t initial_k = 0;
};

struct ChannelSlice {
    size_t start_bit = 0;
    size_t bit_length = 0;
    uint32_t block_size = 0;
    uint8_t residual_mode = 0;
    uint8_t predictor_type = 2; // 0=fixed,1=fir,2=lpc
    bool has_mode_byte = false;
    bool has_control_byte = false;
    uint8_t partition_order = 0;
    int lpc_order = 0;
    uint32_t initial_k = 0;
    std::vector<int16_t> coeffs;
    std::vector<PartitionSlice> partitions;
};

struct BlockSlice {
    ChannelSlice primary;
    ChannelSlice secondary;
    bool has_secondary = false;
    bool mid_side = false;
};

struct BlockDecodeResult {
    bool ok = true;
    size_t index = 0;
    std::vector<int32_t> left;
    std::vector<int32_t> right;
};

inline int32_t rice_unsigned_to_signed(uint32_t u) {
    return static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
}

uint32_t read_rice_unsigned(BitSliceReader& reader, uint32_t k) {
    uint32_t q = 0;
    while (true) {
        uint32_t bit = reader.read_bit();
        if (reader.has_error()) return 0;
        if (bit == 0) break;
        ++q;
    }
    uint32_t remainder = (k > 0) ? reader.read_bits(static_cast<int>(k)) : 0u;
    return (q << k) | remainder;
}

inline uint32_t adapt_k_stateless(uint64_t sum, uint32_t count) {
    if (count == 0) return 0;
    const uint64_t mean = (sum + (count >> 1)) / count;
    uint32_t k = 0;
    while ((1u << k) < mean && k < 31u) {
        ++k;
    }
    return k;
}

std::vector<uint32_t> partition_sizes_for_block(uint32_t size, uint8_t order) {
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
}

void synthesize_lpc(int order,
                    const std::vector<int16_t>& coeffs_q15,
                    const std::vector<int32_t>& residual,
                    std::vector<int32_t>& pcm) {
    LPC lpc(order);
    lpc.restore_from_residual_q15(residual, coeffs_q15, pcm);
}

void synthesize_fixed(int order,
                      const std::vector<int32_t>& residual,
                      std::vector<int32_t>& pcm) {
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
            case 1: pred = pcm[i - 1]; break;
            case 2: pred = 2LL * pcm[i - 1] - pcm[i - 2]; break;
            case 3: pred = 3LL * pcm[i - 1] - 3LL * pcm[i - 2] + pcm[i - 3]; break;
            case 4: pred = 4LL * pcm[i - 1] - 6LL * pcm[i - 2] + 4LL * pcm[i - 3] - pcm[i - 4]; break;
            default: pred = 0; break;
        }
        pcm[i] = static_cast<int32_t>(residual[i] + pred);
    }
}

void synthesize_fir(int taps,
                    const std::vector<int32_t>& residual,
                    std::vector<int32_t>& pcm) {
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
}

void reconstruct_mid_side(std::vector<int32_t>& mid,
                          std::vector<int32_t>& side,
                          std::vector<int32_t>& left,
                          std::vector<int32_t>& right) {
    size_t n = mid.size();
    left.resize(n);
    right.resize(n);
    for (size_t i = 0; i < n; ++i) {
        int32_t m = mid[i];
        int32_t s = side[i];
        int32_t l = m + ((s + (s & 1)) >> 1);
        left[i] = l;
        right[i] = l - s;
    }
}

bool decode_residuals(BitSliceReader& reader,
                      uint32_t sample_count,
                      uint32_t initial_k,
                      uint8_t residual_mode,
                      std::vector<int32_t>* residual_out,
                      size_t write_offset = 0,
                      bool stateless = false,
                      int block_index = -1) {
    if (residual_mode > 2) return false;

    if (block_index >= 0 && e2e_debug_enabled()) {
        LAC_TRACE_LOG("[res-enter] block=" << block_index
                      << " samples=" << sample_count
                      << " mode=" << static_cast<int>(residual_mode)
                      << " k=" << initial_k
                      << " stateless=" << (stateless ? 1 : 0)
                      << " offset=" << write_offset
                      << " bitpos=" << reader.bit_position()
                      << "\n");
    }

    if (residual_out) {
        if (write_offset + sample_count > residual_out->size()) {
            residual_out->resize(write_offset + sample_count, 0);
        }
    }
    auto store = [&](uint32_t idx, int32_t v) {
        if (residual_out) {
            (*residual_out)[write_offset + idx] = v;
        }
    };

    if (sample_count == 0) {
        return true;
    }

    uint32_t current_k = initial_k;
    uint64_t sum_u = 0;

    if (residual_mode == 0) {
        for (uint32_t i = 0; i < sample_count; ++i) {
            uint32_t u = read_rice_unsigned(reader, current_k);
            if (reader.has_error()) return false;
            store(i, rice_unsigned_to_signed(u));
            sum_u += u;
            current_k = stateless
                ? adapt_k_stateless(sum_u, i + 1)
                : Rice::adapt_k(sum_u, i + 1);
        }
        return !reader.has_error();
    } else if (residual_mode == 1) {
        if (block_index >= 0) {
            LAC_DEBUG_LOG("[zr-dec] block=" << block_index
                          << " samples=" << sample_count
                          << " k0=" << initial_k
                          << "\n");
        }
        constexpr uint32_t kTagNormal = 0b00u;
        constexpr uint32_t kTagRun = 0b01u;
        constexpr uint32_t kTagEscape = 0b10u;

        auto zigzag_decode = [](uint32_t u) -> int32_t {
            return static_cast<int32_t>((u >> 1) ^ -static_cast<int32_t>(u & 1u));
        };

        uint32_t idx = 0;
        uint32_t count = 0;
        while (idx < sample_count) {
            const size_t tag_bit_pos = reader.bit_position();
            uint32_t tag_prefix = reader.read_bit();
            if (reader.has_error()) {
                if (block_index >= 0) {
                    std::cerr << "[zr-dec-error] block=" << block_index << " pos=" << idx << " reason=tag-prefix\n";
                }
                return false;
            }
            uint32_t tag = kTagEscape;
            if (tag_prefix == 0) {
                // Distinguish normal vs run when not an escape prefix.
                tag = reader.read_bit();
            } else {
                // Consume the second tag bit for escape tokens to stay aligned.
                (void)reader.read_bit();
            }
            if (reader.has_error()) {
                if (block_index >= 0) {
                    std::cerr << "[zr-dec-error] block=" << block_index << " pos=" << idx << " reason=tag\n";
                }
                return false;
            }
            if (tag == kTagNormal) {
                uint32_t u = read_rice_unsigned(reader, current_k);
                if (reader.has_error() || idx >= sample_count) {
                    if (block_index >= 0) {
                        std::cerr << "[zr-dec-error] block=" << block_index << " pos=" << idx << " reason=normal-read\n";
                    }
                    return false;
                }
                if (block_index >= 0) {
                    const size_t abs_idx = write_offset + idx;
                    LAC_DEBUG_LOG("[zr-dec-token] block=" << block_index
                                  << " idx=" << abs_idx
                                  << " tag=normal"
                                  << " bitpos=" << tag_bit_pos
                                  << " val=" << rice_unsigned_to_signed(u)
                                  << "\n");
                }
                store(idx++, rice_unsigned_to_signed(u));
                sum_u += u;
                ++count;
                current_k = stateless
                    ? adapt_k_stateless(sum_u, count)
                    : Rice::adapt_k(sum_u, count);
            } else if (tag == kTagRun) {
                uint32_t run_len = read_rice_unsigned(reader, Block::ZERO_RUN_LENGTH_K) + Block::ZERO_RUN_MIN_LENGTH;
                if (reader.has_error() || idx + run_len > sample_count) {
                    if (block_index >= 0) {
                        std::cerr << "[zr-dec-error] block=" << block_index << " pos=" << idx << " reason=run-len\n";
                    }
                    return false;
                }
                assert(run_len >= 1);
                if (block_index >= 0) {
                    const size_t abs_idx = write_offset + idx;
                    LAC_DEBUG_LOG("[zr-dec-token] block=" << block_index
                                  << " idx=" << abs_idx
                                  << " tag=run"
                                  << " bitpos=" << tag_bit_pos
                                  << " val=" << run_len
                                  << "\n");
                }
                for (uint32_t j = 0; j < run_len; ++j) {
                    store(idx++, 0);
                    ++count;
                    current_k = stateless
                        ? adapt_k_stateless(sum_u, count)
                        : Rice::adapt_k(sum_u, count);
                }
            } else if (tag == kTagEscape) {
                if (idx >= sample_count) {
                    if (block_index >= 0) {
                        std::cerr << "[zr-dec-error] block=" << block_index << " pos=" << idx << " reason=esc-oob\n";
                    }
                    return false;
                }
                uint32_t zz = reader.read_bits(32);
                if (reader.has_error()) {
                    if (block_index >= 0) {
                        std::cerr << "[zr-dec-error] block=" << block_index << " pos=" << idx << " reason=esc-read\n";
                    }
                    break;
                }
                int32_t value = zigzag_decode(zz);
                if (block_index >= 0) {
                    const size_t abs_idx = write_offset + idx;
                    LAC_DEBUG_LOG("[zr-dec-token] block=" << block_index
                                  << " idx=" << abs_idx
                                  << " tag=escape"
                                  << " bitpos=" << tag_bit_pos
                                  << " val=" << value
                                  << "\n");
                }
                store(idx++, value);
                uint32_t u = static_cast<uint32_t>((value << 1) ^ (value >> 31));
                sum_u += u;
                ++count;
                current_k = stateless
                    ? adapt_k_stateless(sum_u, count)
                    : Rice::adapt_k(sum_u, count);
            } else {
                return false;
            }
        }
        if (idx != sample_count && block_index >= 0) {
        std::cerr << "[zr-dec-error] block=" << block_index
                  << " reason=count-mismatch"
                  << " decoded=" << idx
                  << " expected=" << sample_count
                  << "\n";
        }
        return idx == sample_count;
    } else if (residual_mode == 2) {
        uint32_t idx = 0;
        uint32_t count = 0;
        while (idx < sample_count) {
            size_t tag_pos = reader.bit_position();
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
                value = rice_unsigned_to_signed(u);
                store(idx++, value);
                sum_u += u;
                ++count;
                current_k = stateless
                    ? adapt_k_stateless(sum_u, count)
                    : Rice::adapt_k(sum_u, count);
                continue;
            } else {
                return false;
            }
            uint32_t u = static_cast<uint32_t>((value << 1) ^ (value >> 31));
            store(idx++, value);
            sum_u += u;
            ++count;
            current_k = stateless
                ? adapt_k_stateless(sum_u, count)
                : Rice::adapt_k(sum_u, count);
            if (block_index >= 0 && e2e_debug_enabled()) {
                LAC_TRACE_LOG("[e2e-res] block=" << block_index
                              << " idx=" << (write_offset + idx - 1)
                              << " tag=" << (tag == kBinTagZero ? "bin0" : tag == kBinTagOne ? "bin1" : tag == kBinTagTwo ? "bin2" : "binfb")
                              << " bitpos=" << tag_pos
                              << " val=" << value
                              << " k=" << current_k
                              << "\n");
            }
        }
        return idx == sample_count;
    }

    return false;
}

bool scan_channel(BitSliceReader& reader,
                  uint32_t block_size,
                  ChannelSlice& slice,
                  int block_index,
                  int channel_id) {
    if (block_size == 0 || block_size > Block::MAX_BLOCK_SIZE) {
        if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=block_size\n"); }
        return false;
    }

    BitSliceReader local_reader = reader;
    ChannelSlice temp;
    temp.start_bit = local_reader.bit_position();
    temp.block_size = block_size;
    temp.has_control_byte = true;
    temp.has_mode_byte = true;
    // Header layout per channel: predictor_type (8), predictor_order (8), LPC coeffs (if LPC),
    // residual_control (8), partition metadata ([mode:2|k:5] * partitions), residual bits, byte aligned.
    temp.predictor_type = static_cast<uint8_t>(local_reader.read_bits(8));
    if (local_reader.has_error()) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=predictor-byte\n"); } return false; }
    if (temp.predictor_type > 2) {
        if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=predictor-type val=" << static_cast<int>(temp.predictor_type) << "\n"); }
        return false;
    }

    temp.lpc_order = static_cast<int>(local_reader.read_bits(8));
    if (local_reader.has_error()) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=order-read\n"); } return false; }
    if (temp.predictor_type == kPredictorLpc) {
        if (temp.lpc_order <= 0 || temp.lpc_order > 32 || static_cast<uint32_t>(temp.lpc_order) >= block_size) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=lpc-order-invalid val=" << temp.lpc_order << "\n"); } return false; }
    } else {
        if (temp.lpc_order < 0 || temp.lpc_order > 32) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=fixed-order-invalid val=" << temp.lpc_order << "\n"); } return false; }
    }

    if (temp.predictor_type == kPredictorLpc) {
        temp.coeffs.assign(static_cast<size_t>(temp.lpc_order) + 1, 0);
        for (int i = 1; i <= temp.lpc_order; ++i) {
            temp.coeffs[i] = static_cast<int16_t>(local_reader.read_bits(16));
            if (local_reader.has_error()) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=coeff-read\n"); } return false; }
        }
    } else {
        temp.coeffs.clear();
    }

    const uint8_t control = static_cast<uint8_t>(local_reader.read_bits(8));
    if (local_reader.has_error()) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=control-byte\n"); } return false; }
    const bool partition_flag = (control & Block::PARTITION_FLAG) != 0u;
    const uint8_t control_mode = static_cast<uint8_t>((control >> 5) & 0x03u);
    temp.partition_order = partition_flag
        ? static_cast<uint8_t>((control & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT)
        : 0;
    if (partition_flag && temp.partition_order == 0) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=partition-flag-zero\n"); } return false; }
    if (temp.partition_order > Block::MAX_PARTITION_ORDER) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=partition-order\n"); } return false; }
    if ((temp.partition_order > 0) && ((block_size >> temp.partition_order) < Block::MIN_PARTITION_SIZE)) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=partition-too-small\n"); } return false; }

    const uint32_t partition_count = (temp.partition_order == 0) ? 1u : (1u << temp.partition_order);
    const auto part_sizes = partition_sizes_for_block(block_size, temp.partition_order);
    if (part_sizes.size() != partition_count || part_sizes.back() == 0) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=partition-size\n"); } return false; }

    std::vector<uint8_t> part_modes(partition_count);
    std::vector<uint32_t> part_k(partition_count);
    for (uint32_t i = 0; i < partition_count; ++i) {
        part_modes[i] = static_cast<uint8_t>(local_reader.read_bits(2));
        part_k[i] = local_reader.read_bits(5);
        if (local_reader.has_error()) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=partition-meta\n"); } return false; }
        if (part_modes[i] > 2) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=mode-invalid\n"); } return false; }
    }

    if (e2e_debug_enabled()) {
        LAC_TRACE_LOG("[e2e-hdr] block=" << block_index
                      << " ch=" << channel_id
                      << " predict=" << static_cast<int>(temp.predictor_type)
                      << " order=" << temp.lpc_order
                      << " ctrl=" << static_cast<int>(control)
                      << " mode=" << static_cast<int>(control_mode)
                      << " pflag=" << (partition_flag ? 1 : 0)
                      << " porder=" << static_cast<int>(temp.partition_order)
                      << " pos=" << temp.start_bit
                      << "\n");
        std::ostringstream oss;
        oss << "[e2e-meta] block=" << block_index
            << " ch=" << channel_id
            << " parts=" << partition_count
            << " bits_before_res=" << local_reader.bit_position();
        for (uint32_t i = 0; i < partition_count; ++i) {
            oss << " (" << static_cast<int>(part_modes[i]) << "," << part_k[i] << "," << part_sizes[i] << ")";
        }
        oss << "\n";
        LAC_TRACE_LOG(oss.str());
    }

    const bool stateless = (temp.partition_order > 0);

    if (temp.partition_order == 0) {
        temp.residual_mode = part_modes[0];
        temp.initial_k = part_k[0];
        PartitionSlice ps;
        ps.sample_count = block_size;
        ps.residual_mode = temp.residual_mode;
        ps.initial_k = temp.initial_k;
        ps.start_bit = local_reader.bit_position();
        if (!decode_residuals(local_reader, block_size, temp.initial_k, temp.residual_mode, nullptr, 0, stateless, block_index)) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=decode-residuals\n"); } return false; }
        ps.bit_length = local_reader.bit_position() - ps.start_bit;
        temp.partitions.push_back(ps);
        temp.bit_length = (ps.start_bit + ps.bit_length) - temp.start_bit;
        local_reader.align_to_byte();
        if (local_reader.has_error()) return false;
        if (block_index >= 0 && e2e_debug_enabled()) {
            LAC_TRACE_LOG("[scan-summary] block=" << block_index
                          << " predict=" << static_cast<int>(temp.predictor_type)
                          << " mode=" << static_cast<int>(temp.residual_mode)
                          << " order=" << static_cast<int>(temp.partition_order)
                          << " bitlen=" << temp.bit_length
                          << " part0=" << ps.bit_length
                          << " start=" << temp.start_bit
                          << " size=" << temp.block_size
                          << " ch=" << channel_id
                          << "\n");
        }
        slice = temp;
        reader.copy_from(local_reader);
        if (e2e_debug_enabled()) {
            LAC_TRACE_LOG("[e2e-attempt-done] block=" << block_index
                          << " ch=" << channel_id
                          << " end=" << reader.bit_position()
                          << " addr=" << &reader
                          << "\n");
        }
        return true;
    }

    uint64_t sample_sum = 0;
    for (uint32_t i = 0; i < partition_count; ++i) {
        PartitionSlice ps;
        ps.sample_count = part_sizes[i];
        ps.residual_mode = part_modes[i];
        ps.initial_k = part_k[i];
        ps.start_bit = local_reader.bit_position();
        if (!decode_residuals(local_reader, ps.sample_count, ps.initial_k, ps.residual_mode, nullptr, 0, stateless, block_index)) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan fail block=" << block_index << " reason=decode-partial part=" << i << "\n"); } return false; }
        ps.bit_length = local_reader.bit_position() - ps.start_bit;
        temp.partitions.push_back(ps);
        sample_sum += ps.sample_count;
    }
    if (sample_sum != block_size) return false;

    size_t max_end = temp.start_bit;
    for (const auto& ps : temp.partitions) {
        max_end = std::max(max_end, ps.start_bit + ps.bit_length);
    }
    temp.bit_length = max_end - temp.start_bit;
    local_reader.align_to_byte();
    if (local_reader.has_error()) return false;
    if (block_index >= 0) {
        if (e2e_debug_enabled()) {
            for (size_t idx = 0; idx < temp.partitions.size(); ++idx) {
                const auto& ps = temp.partitions[idx];
                LAC_TRACE_LOG("[e2e-part] block=" << block_index
                              << " part=" << idx
                              << " predict=" << static_cast<int>(temp.predictor_type)
                              << " order=" << static_cast<int>(temp.partition_order)
                              << " mode=" << static_cast<int>(ps.residual_mode)
                              << " k=" << ps.initial_k
                              << " start=" << ps.start_bit
                              << " len=" << ps.bit_length
                              << " samples=" << ps.sample_count
                              << "\n");
            }
            LAC_TRACE_LOG("[scan-summary] block=" << block_index
                          << " predict=" << static_cast<int>(temp.predictor_type)
                          << " mode=partitions"
                          << " order=" << static_cast<int>(temp.partition_order)
                          << " bitlen=" << temp.bit_length
                          << " parts=" << temp.partitions.size()
                          << " start=" << temp.start_bit
                          << " ch=" << channel_id
                          << "\n");
        }
    }
    slice = temp;
    reader.copy_from(local_reader);
    if (e2e_debug_enabled()) {
        LAC_TRACE_LOG("[e2e-attempt-done] block=" << block_index
                      << " ch=" << channel_id
                      << " end=" << reader.bit_position()
                      << " addr=" << &reader
                      << "\n");
    }
    return true;
}

bool decode_channel_slice(const uint8_t* payload,
                          const ChannelSlice& slice,
                          std::vector<int32_t>& pcm,
                          int block_index) {
    if (slice.partitions.empty()) {
        std::cerr << "[zr-dec-error] block=" << block_index << " reason=empty-slice\n";
        return false;
    }
    const size_t expected_bits = slice.bit_length;
    const size_t slice_start = slice.start_bit;
    std::vector<int32_t> residual(slice.block_size, 0);
    size_t sample_offset = 0;
    size_t max_end_bit = slice.start_bit;

    for (const auto& part : slice.partitions) {
        BitSliceReader part_reader(payload, part.start_bit, part.bit_length);
        if (!decode_residuals(part_reader, part.sample_count, part.initial_k, part.residual_mode, &residual, sample_offset, slice.partition_order > 0, block_index)) {
            std::cerr << "[zr-dec-error] block=" << block_index
                      << " reason=part-fail"
                      << " start=" << part.start_bit
                      << " len=" << part.bit_length
                      << " samples=" << part.sample_count
                      << "\n";
            return false;
        }
        assert(part_reader.bits_remaining() == 0);
        sample_offset += part.sample_count;
        max_end_bit = std::max(max_end_bit, part.start_bit + part.bit_length);
    }
    if (sample_offset != slice.block_size) {
        std::cerr << "[zr-dec-error] block=" << block_index
                  << " reason=sample-mismatch"
                  << " got=" << sample_offset
                  << " expected=" << slice.block_size
                  << "\n";
        return false;
    }
    if ((max_end_bit - slice.start_bit) != slice.bit_length) {
            std::cerr << "[zr-dec-error] block=" << block_index
                      << " reason=slice-bits"
                      << " expected=" << slice.bit_length
                      << " actual=" << (max_end_bit - slice.start_bit)
                      << " parts=" << slice.partitions.size()
                  << " start=" << slice.start_bit
                  << "\n";
        return false;
    }

    uint8_t predictor_type = slice.predictor_type;
    if (predictor_type == kPredictorLpc && (slice.lpc_order <= 0 || slice.coeffs.empty())) {
        predictor_type = kPredictorFixed;
    }

    if (predictor_type == kPredictorFixed) {
        synthesize_fixed(slice.lpc_order, residual, pcm);
    } else if (predictor_type == kPredictorFir) {
        synthesize_fir(slice.lpc_order, residual, pcm);
    } else {
        synthesize_lpc(slice.lpc_order, slice.coeffs, residual, pcm);
    }
    (void)expected_bits;
    assert(pcm.size() == slice.block_size);
    return pcm.size() == slice.block_size;
}

} // namespace

Decoder::Decoder(ThreadCollector* collector)
    : collector(collector) {}

bool Decoder::decode(const uint8_t* data,
                     size_t size,
                     std::vector<int32_t>& left,
                     std::vector<int32_t>& right,
                     FrameHeader* out_header) {
    left.clear();
    right.clear();

    FrameHeader hdr;
    size_t header_bytes = 0;
    if (!FrameHeader::parse(data, size, hdr, header_bytes)) {
    if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] header parse failed\n"); }
        return false;
    }

    if (size < header_bytes) return false;
    const uint8_t* payload = data + header_bytes;
    size_t payload_bytes = size - header_bytes;
    BitSliceReader reader(payload, 0, payload_bytes * 8);

    if (reader.bits_remaining() < 32) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] not enough bits for block_count\n"); } return false; }
    uint32_t block_count = reader.read_bits(32);
    if (reader.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] invalid block_count=" << block_count << "\n"); } return false; }
    if (reader.bits_remaining() < static_cast<uint64_t>(block_count) * 32ULL) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] not enough bits for block sizes\n"); } return false; }

    std::vector<uint32_t> block_sizes(block_count);
    uint64_t total_samples = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        uint32_t sz = reader.read_bits(32);
        if (reader.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] invalid block size index=" << i << " sz=" << sz << "\n"); } return false; }
        total_samples += sz;
        if (total_samples > MAX_TOTAL_SAMPLES) return false;
        block_sizes[i] = sz;
    }

    const bool isStereo = (hdr.channels == 2);
    const bool perBlockStereo = isStereo && (hdr.stereo_mode == 2);
    const bool forceMidSide = isStereo && (hdr.stereo_mode == 1);

    std::vector<size_t> blockOffsets(block_count);
    size_t running = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        blockOffsets[i] = running;
        running += block_sizes[i];
    }

    left.assign(running, 0);
    if (isStereo) {
        right.assign(running, 0);
    } else {
        right.clear();
    }

    std::vector<BlockSlice> slices(block_count);
    for (uint32_t i = 0; i < block_count; ++i) {
        if (perBlockStereo) {
            uint32_t mode_flag = reader.read_bits(8);
            if (reader.has_error()) return false;
            slices[i].mid_side = (mode_flag != 0);
        } else if (forceMidSide) {
            slices[i].mid_side = true;
        } else {
            slices[i].mid_side = false;
        }
        if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] pre-scan block=" << i << " ch=0 pos=" << reader.bit_position() << " size=" << block_sizes[i] << " addr=" << &reader << "\n"); }
        if (!scan_channel(reader, block_sizes[i], slices[i].primary, static_cast<int>(i), 0)) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan primary fail block=" << i << "\n"); } return false; }
        size_t primary_end = slices[i].primary.start_bit + slices[i].primary.bit_length;
        primary_end += (8u - (primary_end & 7u)) & 7u;
        reader.set_position(primary_end);
        if (isStereo) {
            slices[i].has_secondary = true;
            if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] pre-scan block=" << i << " ch=1 pos=" << reader.bit_position() << " size=" << block_sizes[i] << " addr=" << &reader << "\n"); }
            if (!scan_channel(reader, block_sizes[i], slices[i].secondary, static_cast<int>(i), 1)) { if (e2e_debug_enabled()) { LAC_TRACE_LOG("[e2e] scan secondary fail block=" << i << "\n"); } return false; }
            size_t secondary_end = slices[i].secondary.start_bit + slices[i].secondary.bit_length;
            secondary_end += (8u - (secondary_end & 7u)) & 7u;
            reader.set_position(secondary_end);
        }
        if (e2e_debug_enabled()) {
            LAC_TRACE_LOG("[e2e] post-block=" << i << " pos=" << reader.bit_position() << " addr=" << &reader << "\n");
        }
    }

    const size_t debug_blocks = std::min<size_t>(3, slices.size());
    for (size_t i = 0; i < debug_blocks; ++i) {
        const auto& s = slices[i].primary;
        if (!s.partitions.empty()) {
            LAC_TRACE_LOG("[slice] block=" << i
                          << " start=" << s.start_bit
                          << " len=" << s.bit_length
                          << " parts=" << s.partitions.size()
                          << " p0_start=" << s.partitions[0].start_bit
                          << " p0_len=" << s.partitions[0].bit_length
                          << "\n");
        }
        if (slices[i].has_secondary && !slices[i].secondary.partitions.empty()) {
            const auto& sec = slices[i].secondary;
            LAC_TRACE_LOG("[slice] block=" << i
                          << " secondary start=" << sec.start_bit
                          << " len=" << sec.bit_length
                          << " parts=" << sec.partitions.size()
                          << " p0_start=" << sec.partitions[0].start_bit
                          << " p0_len=" << sec.partitions[0].bit_length
                          << "\n");
        }
    }

    auto launch_policy = std::thread::hardware_concurrency() > 1
        ? std::launch::async
        : std::launch::deferred;

    std::vector<std::future<BlockDecodeResult>> blockTasks;
    blockTasks.reserve(block_count);

    for (uint32_t idx = 0; idx < block_count; ++idx) {
        blockTasks.emplace_back(std::async(
            launch_policy,
            [this,
             idx,
             &slices,
             isStereo,
             perBlockStereo,
             forceMidSide,
             payload]() -> BlockDecodeResult {
                if (this->collector) {
                    this->collector->record(std::this_thread::get_id());
                }

                BlockDecodeResult result;
                result.index = idx;

                if (e2e_debug_enabled()) {
                    LAC_TRACE_LOG("[e2e-block] block=" << idx
                                  << " predictA=" << static_cast<int>(slices[idx].primary.predictor_type)
                                  << " predictB=" << (slices[idx].has_secondary ? static_cast<int>(slices[idx].secondary.predictor_type) : -1)
                                  << " stereo=" << (perBlockStereo ? (slices[idx].mid_side ? "MS" : "LR") : (forceMidSide ? "MS" : "LR"))
                                  << "\n");
                }

                std::vector<int32_t> primary_pcm;
                if (!decode_channel_slice(payload, slices[idx].primary, primary_pcm, static_cast<int>(idx))) {
                    std::cerr << "[zr-dec-error] block=" << idx << " reason=primary-fail\n";
                    result.ok = false;
                    return result;
                }

                std::vector<int32_t> secondary_pcm;
                if (slices[idx].has_secondary) {
                    if (!decode_channel_slice(payload, slices[idx].secondary, secondary_pcm, static_cast<int>(idx))) {
                        std::cerr << "[zr-dec-error] block=" << idx << " reason=secondary-fail\n";
                        result.ok = false;
                        return result;
                    }
                }

                if (!isStereo) {
                    result.left = std::move(primary_pcm);
                } else {
                    bool mid_side_block = perBlockStereo ? slices[idx].mid_side : forceMidSide;
                    if (!mid_side_block) {
                        result.left = std::move(primary_pcm);
                        result.right = std::move(secondary_pcm);
                    } else {
                        std::vector<int32_t> left_pcm;
                        std::vector<int32_t> right_pcm;
                        reconstruct_mid_side(primary_pcm, secondary_pcm, left_pcm, right_pcm);
                        result.left = std::move(left_pcm);
                        result.right = std::move(right_pcm);
                    }
                }
                return result;
            }
        ));
    }

    for (auto& futureResult : blockTasks) {
        BlockDecodeResult result = futureResult.get();
        if (!result.ok) {
            std::cerr << "[decode-error] block=" << result.index << "\n";
            left.clear();
            right.clear();
            return false;
        }
        size_t offset = blockOffsets[result.index];
        std::copy(result.left.begin(), result.left.end(), left.begin() + offset);
        if (isStereo) {
            std::copy(result.right.begin(), result.right.end(), right.begin() + offset);
        }
    }

    if (hdr.channels == 2 && right.size() != left.size()) {
        left.clear();
        right.clear();
        return false;
    }

    if (out_header) {
        *out_header = hdr;
    }

    return true;
}

} // namespace LAC
