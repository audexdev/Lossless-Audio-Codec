#include "decoder.hpp"

namespace LAC {

namespace {
constexpr uint32_t MAX_BLOCK_COUNT = 10'000'000;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz
}

Decoder::Decoder() {}

bool Decoder::decode(const uint8_t* data, size_t size, std::vector<int32_t>& left, std::vector<int32_t>& right, FrameHeader* out_header) {
    left.clear();
    right.clear();

    FrameHeader hdr;
    size_t header_bytes = 0;
    if (!FrameHeader::parse(data, size, hdr, header_bytes))
        return false;

    BitReader br(data + header_bytes, size - header_bytes);

    if (br.bits_remaining() < 32) return false;
    uint32_t block_count = br.read_bits(32);
    if (br.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) return false;
    if (static_cast<uint64_t>(block_count) * 32ULL > br.bits_remaining()) return false;

    std::vector<uint32_t> block_sizes(block_count);
    uint64_t total_samples = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        if (br.bits_remaining() < 32) return false;
        uint32_t sz = br.read_bits(32);
        if (br.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE) return false;
        total_samples += sz;
        if (total_samples > MAX_TOTAL_SAMPLES) return false;
        block_sizes[i] = sz;
    }

    Block::Decoder blockDec;

    std::vector<int32_t> tempA;
    std::vector<int32_t> tempB;

    if (hdr.channels == 1) {
        for (uint32_t size_entry : block_sizes) {
            tempA.clear();
            if (!blockDec.decode(br, size_entry, tempA)) {
                left.clear();
                right.clear();
                return false;
            }
            left.insert(left.end(), tempA.begin(), tempA.end());
        }
    } else {
        if (hdr.stereo_mode == 0) {
            for (uint32_t size_entry : block_sizes) {
                tempA.clear();
                tempB.clear();
                if (!blockDec.decode(br, size_entry, tempA) ||
                    !blockDec.decode(br, size_entry, tempB)) {
                    left.clear();
                    right.clear();
                    return false;
                }
                left.insert(left.end(), tempA.begin(), tempA.end());
                right.insert(right.end(), tempB.begin(), tempB.end());
            }
        } else if (hdr.stereo_mode == 1) {
            for (uint32_t size_entry : block_sizes) {
                tempA.clear();
                tempB.clear();
                if (!blockDec.decode(br, size_entry, tempA) ||
                    !blockDec.decode(br, size_entry, tempB)) {
                    left.clear();
                    right.clear();
                    return false;
                }
                size_t n = tempA.size();
                size_t base = left.size();
                left.resize(base + n);
                right.resize(base + n);
                int32_t* lptr = left.data() + base;
                int32_t* rptr = right.data() + base;
                for (size_t i = 0; i < n; ++i) {
                    int32_t m = tempA[i];
                    int32_t s = tempB[i];
                    int32_t l = m + ((s + (s & 1)) >> 1);
                    lptr[i] = l;
                    rptr[i] = l - s;
                }
            }
        } else {
            return false;
        }
    }

    if (br.has_error()) {
        left.clear();
        right.clear();
        return false;
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
