#include "decoder.hpp"
#include <algorithm>
#include <future>
#include <thread>
#include <cassert>
#include <iostream>
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"

namespace LAC {
namespace {

constexpr uint32_t MAX_BLOCK_COUNT = 10'000'000;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz

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
    if (residual_mode > 1) return false;

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
    }

#ifdef LAC_ENABLE_ZERORUN
    if (residual_mode == 1) {
        if (block_index >= 0) {
            std::cerr << "[zr-dec] block=" << block_index
                      << " samples=" << sample_count
                      << " k0=" << initial_k
                      << "\n";
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
                    std::cerr << "[zr-dec-token] block=" << block_index
                              << " idx=" << abs_idx
                              << " tag=normal"
                              << " bitpos=" << tag_bit_pos
                              << " val=" << rice_unsigned_to_signed(u)
                              << "\n";
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
                    std::cerr << "[zr-dec-token] block=" << block_index
                              << " idx=" << abs_idx
                              << " tag=run"
                              << " bitpos=" << tag_bit_pos
                              << " val=" << run_len
                              << "\n";
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
                    std::cerr << "[zr-dec-token] block=" << block_index
                              << " idx=" << abs_idx
                              << " tag=escape"
                              << " bitpos=" << tag_bit_pos
                              << " val=" << value
                              << "\n";
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
    }
#endif

    return false;
}

bool scan_channel(BitSliceReader& reader,
                  uint32_t block_size,
                  ChannelSlice& slice,
                  int block_index) {
    auto attempt = [&](BitSliceReader local_reader, bool has_control_byte) -> bool {
        if (block_size == 0 || block_size > Block::MAX_BLOCK_SIZE) return false;

        ChannelSlice temp;
        temp.start_bit = local_reader.bit_position();
        temp.block_size = block_size;
        temp.has_control_byte = has_control_byte;

        if (has_control_byte) {
            const uint8_t control = static_cast<uint8_t>(local_reader.read_bits(8));
            if (local_reader.has_error()) return false;
            const bool partition_flag = (control & Block::PARTITION_FLAG) != 0u;
            temp.has_mode_byte = !partition_flag;
            if (partition_flag) {
                temp.partition_order = static_cast<uint8_t>((control & Block::PARTITION_ORDER_MASK) >> Block::PARTITION_ORDER_SHIFT);
                if (temp.partition_order > Block::MAX_PARTITION_ORDER) return false;
                if ((block_size >> temp.partition_order) < Block::MIN_PARTITION_SIZE) return false;
            } else {
                temp.partition_order = 0;
                temp.residual_mode = control;
                if (temp.residual_mode > 1) return false;
#ifndef LAC_ENABLE_ZERORUN
                if (temp.residual_mode == 1) return false;
#endif
            }
        }

        temp.lpc_order = static_cast<int>(local_reader.read_bits(8));
        if (local_reader.has_error()) return false;
        if (temp.lpc_order <= 0 || temp.lpc_order > 32 || static_cast<uint32_t>(temp.lpc_order) >= block_size) return false;

        if (temp.partition_order == 0) {
            temp.initial_k = local_reader.read_bits(5);
            if (local_reader.has_error()) return false;
        }

        temp.coeffs.assign(static_cast<size_t>(temp.lpc_order) + 1, 0);
        for (int i = 1; i <= temp.lpc_order; ++i) {
            temp.coeffs[i] = static_cast<int16_t>(local_reader.read_bits(16));
            if (local_reader.has_error()) return false;
        }

        const bool stateless = (temp.partition_order > 0);

        if (temp.partition_order == 0) {
            PartitionSlice ps;
            ps.sample_count = block_size;
            ps.residual_mode = temp.residual_mode;
            ps.initial_k = temp.initial_k;
            ps.start_bit = local_reader.bit_position();
            if (!decode_residuals(local_reader, block_size, temp.initial_k, temp.residual_mode, nullptr, 0, stateless, block_index)) return false;
            ps.bit_length = local_reader.bit_position() - ps.start_bit;
            temp.partitions.push_back(ps);
            temp.bit_length = (ps.start_bit + ps.bit_length) - temp.start_bit;
            local_reader.align_to_byte();
            if (local_reader.has_error()) return false;
            if (temp.residual_mode == 1 && block_index >= 0) {
                std::cerr << "[zr-scan] block=" << block_index
                          << " start=" << temp.start_bit
                          << " len=" << temp.bit_length
                          << " samples=" << ps.sample_count
                          << "\n";
            }
            if (block_index >= 0) {
                std::cerr << "[scan-summary] block=" << block_index
                          << " mode=" << static_cast<int>(temp.residual_mode)
                          << " order=" << static_cast<int>(temp.partition_order)
                          << " bitlen=" << temp.bit_length
                          << " part0=" << ps.bit_length
                          << " start=" << temp.start_bit
                          << "\n";
            }
            slice = temp;
            reader = local_reader;
            return true;
        }

        const uint32_t expected_parts = 1u << temp.partition_order;
        const auto part_sizes = partition_sizes_for_block(block_size, temp.partition_order);
        if (part_sizes.size() != expected_parts || part_sizes.back() == 0) return false;

        std::vector<uint8_t> part_modes(expected_parts);
        std::vector<uint32_t> part_k(expected_parts);
        for (uint32_t i = 0; i < expected_parts; ++i) {
            part_modes[i] = static_cast<uint8_t>(local_reader.read_bit());
            part_k[i] = local_reader.read_bits(5);
            if (local_reader.has_error()) return false;
#ifndef LAC_ENABLE_ZERORUN
            if (part_modes[i] == 1) return false;
#endif
        }

        uint64_t sample_sum = 0;
        for (uint32_t i = 0; i < expected_parts; ++i) {
            PartitionSlice ps;
            ps.sample_count = part_sizes[i];
            ps.residual_mode = part_modes[i];
            ps.initial_k = part_k[i];
                ps.start_bit = local_reader.bit_position();
                if (!decode_residuals(local_reader, ps.sample_count, ps.initial_k, ps.residual_mode, nullptr, 0, stateless, block_index)) return false;
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
        for (const auto& ps : temp.partitions) {
            if (ps.residual_mode == 1) {
                std::cerr << "[zr-scan] block=" << block_index
                          << " start=" << ps.start_bit
                          << " len=" << ps.bit_length
                          << "\n";
            }
        }
        std::cerr << "[scan-summary] block=" << block_index
                  << " mode=partitions"
                  << " order=" << static_cast<int>(temp.partition_order)
                  << " bitlen=" << temp.bit_length
                  << " parts=" << temp.partitions.size()
                  << " start=" << temp.start_bit
                  << "\n";
    }
    slice = temp;
    reader = local_reader;
    return true;
    };

    BitSliceReader with_control = reader;
    if (attempt(with_control, true)) return true;

    BitSliceReader legacy = reader;
    if (attempt(legacy, false)) return true;
    return false;
}

bool decode_channel_slice(const uint8_t* payload,
                          const ChannelSlice& slice,
                          std::vector<int32_t>& pcm,
                          int block_index) {
    if (slice.partitions.empty() || slice.coeffs.empty()) {
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

    synthesize_lpc(slice.lpc_order, slice.coeffs, residual, pcm);
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
        return false;
    }

    if (size < header_bytes) return false;
    const uint8_t* payload = data + header_bytes;
    size_t payload_bytes = size - header_bytes;
    BitSliceReader reader(payload, 0, payload_bytes * 8);

    if (reader.bits_remaining() < 32) return false;
    uint32_t block_count = reader.read_bits(32);
    if (reader.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) return false;
    if (reader.bits_remaining() < static_cast<uint64_t>(block_count) * 32ULL) return false;

    std::vector<uint32_t> block_sizes(block_count);
    uint64_t total_samples = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        uint32_t sz = reader.read_bits(32);
        if (reader.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE) return false;
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

        if (!scan_channel(reader, block_sizes[i], slices[i].primary, static_cast<int>(i))) return false;
        if (isStereo) {
            slices[i].has_secondary = true;
            if (!scan_channel(reader, block_sizes[i], slices[i].secondary, static_cast<int>(i))) return false;
        }
    }

    const size_t debug_blocks = std::min<size_t>(3, slices.size());
    for (size_t i = 0; i < debug_blocks; ++i) {
        const auto& s = slices[i].primary;
        if (!s.partitions.empty()) {
            std::cerr << "[slice] block=" << i
                      << " start=" << s.start_bit
                      << " len=" << s.bit_length
                      << " parts=" << s.partitions.size()
                      << " p0_start=" << s.partitions[0].start_bit
                      << " p0_len=" << s.partitions[0].bit_length
                      << "\n";
        }
        if (slices[i].has_secondary && !slices[i].secondary.partitions.empty()) {
            const auto& sec = slices[i].secondary;
            std::cerr << "[slice] block=" << i
                      << " secondary start=" << sec.start_bit
                      << " len=" << sec.bit_length
                      << " parts=" << sec.partitions.size()
                      << " p0_start=" << sec.partitions[0].start_bit
                      << " p0_len=" << sec.partitions[0].bit_length
                      << "\n";
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
