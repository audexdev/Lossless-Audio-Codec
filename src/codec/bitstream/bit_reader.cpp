#include "bit_reader.hpp"
#include <algorithm>

namespace {
inline uint32_t low_bits_mask(int bits) {
    if (bits <= 0) return 0u;
    if (bits >= 32) return 0xFFFFFFFFu;
    return (1u << bits) - 1u;
}
} // namespace

BitReader::BitReader(const uint8_t* data, size_t size) : data(data), size(size), byte_pos(0), bit_pos(0), error(false) {}

BitReader::BitReader(const std::vector<uint8_t>& buf) : data(buf.data()), size(buf.size()), byte_pos(0), bit_pos(0), error(false) {}

void BitReader::mark_error() {
    this->error = true;
    this->byte_pos = this->size;
    this->bit_pos = 0;
}

uint32_t BitReader::read_bit() {
    if (this->byte_pos >= this->size) {
        this->mark_error();
        return 0;
    }

    uint8_t byte = this->data[this->byte_pos];
    int shift = 7 - this->bit_pos;
    uint32_t bit = (byte >> shift) & 1;

    this->bit_pos++;
    if (this->bit_pos == 8) {
        this->bit_pos = 0;
        this->byte_pos++;
    }

    return bit;
}

uint32_t BitReader::read_bits(int nbits) {
    if (nbits <= 0) return 0;

    if (!this->error && static_cast<size_t>(nbits) <= this->bits_remaining()) {
        size_t bit_index = this->byte_pos * 8 + static_cast<size_t>(this->bit_pos);
        int remaining = nbits;
        uint32_t value = 0;

        while (remaining > 0) {
            const size_t byte_index = bit_index / 8;
            const int bit_offset = static_cast<int>(bit_index % 8);

            uint64_t window = 0;
            int window_bits = 0;
            size_t i = byte_index;
            while (window_bits < 64 && i < this->size) {
                window = (window << 8) | this->data[i++];
                window_bits += 8;
            }

            const int usable = window_bits - bit_offset;
            const int take = std::min(remaining, std::min(usable, 24));
            const int shift = usable - take;
            const uint32_t chunk = static_cast<uint32_t>((window >> shift) & low_bits_mask(take));

            value = static_cast<uint32_t>((static_cast<uint64_t>(value) << take) | chunk);
            bit_index += static_cast<size_t>(take);
            remaining -= take;
        }

        this->byte_pos = bit_index / 8;
        this->bit_pos = static_cast<int>(bit_index % 8);
        return value;
    }

    uint32_t value = 0;
    for (int i = 0; i < nbits; ++i) {
        value = (value << 1) | this->read_bit();
        if (this->error) break;
    }

    return value;
}

void BitReader::align_to_byte() {
    if (this->bit_pos == 0) return;
    this->bit_pos = 0;
    this->byte_pos++;
}

bool BitReader::eof() const {
    return this->byte_pos >= this->size;
}

bool BitReader::has_error() const {
    return this->error;
}

size_t BitReader::bits_remaining() const {
    if (this->error) return 0;
    size_t bits = (this->size - this->byte_pos) * 8;
    if (this->bit_pos > 0) {
        bits -= this->bit_pos;
    }
    return bits;
}
