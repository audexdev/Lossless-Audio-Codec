#include "bit_reader.hpp"

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
