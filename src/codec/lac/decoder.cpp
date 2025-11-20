#include "decoder.hpp"
#include <algorithm>
#include <future>
#include <thread>
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

struct ChannelSlice {
    size_t start_bit = 0;
    size_t bit_length = 0;
    uint32_t block_size = 0;
};

struct BlockSlice {
    ChannelSlice primary;
    ChannelSlice secondary;
    bool has_secondary = false;
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

bool scan_channel(BitSliceReader& reader,
                  uint32_t block_size,
                  ChannelSlice& slice) {
    if (block_size == 0 || block_size > Block::MAX_BLOCK_SIZE) return false;
    size_t start = reader.bit_position();

    int order = static_cast<int>(reader.read_bits(8));
    if (reader.has_error()) return false;
    if (order <= 0 || order > 32 || static_cast<uint32_t>(order) >= block_size) return false;

    uint32_t initial_k = reader.read_bits(5);
    if (reader.has_error()) return false;
    uint32_t current_k = initial_k;

    for (int i = 1; i <= order; ++i) {
        reader.read_bits(16);
        if (reader.has_error()) return false;
    }

    uint64_t sum_u = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < block_size; ++i) {
        uint32_t u = read_rice_unsigned(reader, current_k);
        if (reader.has_error()) return false;
        sum_u += u;
        ++count;
        current_k = Rice::adapt_k(sum_u, count);
    }

    reader.align_to_byte();
    if (reader.has_error()) return false;

    size_t end = reader.bit_position();
    if (end < start) return false;

    slice.start_bit = start;
    slice.bit_length = end - start;
    slice.block_size = block_size;
    return true;
}

bool decode_channel_slice(const uint8_t* payload,
                          const ChannelSlice& slice,
                          std::vector<int32_t>& pcm) {
    BitSliceReader reader(payload, slice.start_bit, slice.bit_length);

    int order = static_cast<int>(reader.read_bits(8));
    if (reader.has_error()) return false;
    if (order <= 0 || order > 32 || static_cast<uint32_t>(order) >= slice.block_size) return false;

    uint32_t initial_k = reader.read_bits(5);
    if (reader.has_error()) return false;
    uint32_t current_k = initial_k;

    std::vector<int16_t> coeffs(order + 1);
    coeffs[0] = 0;
    for (int i = 1; i <= order; ++i) {
        coeffs[i] = static_cast<int16_t>(reader.read_bits(16));
        if (reader.has_error()) return false;
    }

    std::vector<int32_t> residual(slice.block_size);
    uint64_t sum_u = 0;
    for (uint32_t i = 0; i < slice.block_size; ++i) {
        uint32_t u = read_rice_unsigned(reader, current_k);
        if (reader.has_error()) return false;
        residual[i] = rice_unsigned_to_signed(u);
        sum_u += u;
        current_k = Rice::adapt_k(sum_u, i + 1);
    }

    reader.align_to_byte();
    if (reader.has_error()) return false;

    synthesize_lpc(order, coeffs, residual, pcm);
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
    const bool useMidSide = isStereo && (hdr.stereo_mode == 1);

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
        if (!scan_channel(reader, block_sizes[i], slices[i].primary)) {
            return false;
        }
        if (isStereo) {
            slices[i].has_secondary = true;
            if (!scan_channel(reader, block_sizes[i], slices[i].secondary)) {
                return false;
            }
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
             &block_sizes,
             isStereo,
             useMidSide,
             payload]() -> BlockDecodeResult {
                if (this->collector) {
                    this->collector->record(std::this_thread::get_id());
                }

                BlockDecodeResult result;
                result.index = idx;

                std::vector<int32_t> primary_pcm;
                if (!decode_channel_slice(payload, slices[idx].primary, primary_pcm)) {
                    result.ok = false;
                    return result;
                }

                std::vector<int32_t> secondary_pcm;
                if (slices[idx].has_secondary) {
                    if (!decode_channel_slice(payload, slices[idx].secondary, secondary_pcm)) {
                        result.ok = false;
                        return result;
                    }
                }

                if (!isStereo) {
                    result.left = std::move(primary_pcm);
                } else if (!useMidSide) {
                    result.left = std::move(primary_pcm);
                    result.right = std::move(secondary_pcm);
                } else {
                    std::vector<int32_t> left_pcm;
                    std::vector<int32_t> right_pcm;
                    reconstruct_mid_side(primary_pcm, secondary_pcm, left_pcm, right_pcm);
                    result.left = std::move(left_pcm);
                    result.right = std::move(right_pcm);
                }
                return result;
            }
        ));
    }

    for (auto& futureResult : blockTasks) {
        BlockDecodeResult result = futureResult.get();
        if (!result.ok) {
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
