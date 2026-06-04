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
        uint32_t value = 0;
        int remaining = nbits;
        while (remaining > 0) {
            const int available = 8 - this->bit_pos;
            const int take = std::min(remaining, available);
            const int shift = available - take;
            const uint32_t chunk =
                (static_cast<uint32_t>(this->data[this->byte_pos]) >> shift) &
                low_bits_mask(take);
            value = static_cast<uint32_t>((value << take) | chunk);
            remaining -= take;
            this->bit_pos += take;
            if (this->bit_pos == 8) {
                this->bit_pos = 0;
                ++this->byte_pos;
            }
        }
        return value;
    }

    uint32_t value = 0;
    for (int i = 0; i < nbits; ++i) {
        value = (value << 1) | this->read_bit();
        if (this->error) break;
    }

    return value;
}

bool BitReader::read_unary_ones(uint32_t max_ones, uint32_t& ones) {
    ones = 0;
    while (this->byte_pos < this->size) {
        if (this->bit_pos == 0 && this->data[this->byte_pos] == 0xFFu) {
            if (max_ones - ones < 8u) return false;
            ones += 8u;
            ++this->byte_pos;
            continue;
        }

        const uint32_t bit = this->read_bit();
        if (this->error) return false;
        if (bit == 0u) return true;
        if (ones == max_ones) return false;
        ++ones;
    }

    this->mark_error();
    return false;
}

void BitReader::align_to_byte() {
    if (this->bit_pos == 0) return;
    this->bit_pos = 0;
    this->byte_pos++;
}

bool BitReader::consume_zero_padding_to_byte() {
    while (this->bit_pos != 0) {
        if (this->read_bit() != 0u || this->error) return false;
    }
    return true;
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
