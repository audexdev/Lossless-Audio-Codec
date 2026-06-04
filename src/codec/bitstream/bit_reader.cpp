#include "bit_reader.hpp"
#include <algorithm>
#include <bit>

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

    if (this->error || static_cast<size_t>(nbits) > this->bits_remaining()) {
        this->mark_error();
        return 0;
    }

    if (this->bit_pos == 0) {
        if (nbits == 8) {
            return this->data[this->byte_pos++];
        }
        if (nbits == 16) {
            const uint32_t value =
                (static_cast<uint32_t>(this->data[this->byte_pos]) << 8) |
                static_cast<uint32_t>(this->data[this->byte_pos + 1]);
            this->byte_pos += 2;
            return value;
        }
        if (nbits == 32) {
            const uint32_t value =
                (static_cast<uint32_t>(this->data[this->byte_pos]) << 24) |
                (static_cast<uint32_t>(this->data[this->byte_pos + 1]) << 16) |
                (static_cast<uint32_t>(this->data[this->byte_pos + 2]) << 8) |
                static_cast<uint32_t>(this->data[this->byte_pos + 3]);
            this->byte_pos += 4;
            return value;
        }
    }

    const int available = 8 - this->bit_pos;
    if (nbits <= available) {
        const int shift = available - nbits;
        const uint32_t value =
            (static_cast<uint32_t>(this->data[this->byte_pos]) >> shift) &
            low_bits_mask(nbits);
        this->bit_pos += nbits;
        if (this->bit_pos == 8) {
            this->bit_pos = 0;
            ++this->byte_pos;
        }
        return value;
    }

    uint32_t value = 0;
    int remaining = nbits;
    while (remaining > 0) {
        const int bits_available = 8 - this->bit_pos;
        const int take = std::min(remaining, bits_available);
        const int shift = bits_available - take;
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

bool BitReader::read_unary_ones(uint32_t max_ones, uint32_t& ones) {
    ones = 0;
    while (this->byte_pos < this->size) {
        const int available = 8 - this->bit_pos;
        const uint32_t shifted =
            (static_cast<uint32_t>(this->data[this->byte_pos]) << this->bit_pos) & 0xFFu;
        const uint32_t run = std::min<uint32_t>(
            static_cast<uint32_t>(available),
            std::countl_one(static_cast<uint32_t>(shifted << 24)));

        if (max_ones - ones < run) {
            return false;
        }
        ones += run;
        this->bit_pos += static_cast<int>(run);
        if (this->bit_pos == 8) {
            this->bit_pos = 0;
            ++this->byte_pos;
        }

        if (run < static_cast<uint32_t>(available)) {
            ++this->bit_pos; // consume unary terminator zero
            if (this->bit_pos == 8) {
                this->bit_pos = 0;
                ++this->byte_pos;
            }
            return true;
        }
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
