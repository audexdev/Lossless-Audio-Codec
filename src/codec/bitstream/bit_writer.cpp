#include "bit_writer.hpp"

BitWriter::BitWriter() : current_byte(0), bit_pos(0) {}

void BitWriter::write_bit(uint32_t bit) {
    if (bit > 1) bit = 1;

    int shift = 7 - this->bit_pos;
    this->current_byte |= (bit << shift);
    this->bit_pos++;

    if (this->bit_pos == 8) {
        this->buffer.push_back(this->current_byte);
        this->current_byte = 0;
        this->bit_pos = 0;
    }
}

void BitWriter::write_bits(uint32_t value, int nbits) {
    for (int i = nbits - 1; i >= 0; --i) {
        uint32_t bit = (value >> i) & 1;
        this->write_bit(bit);
    }
}

void BitWriter::flush_to_byte() {
    if (this->bit_pos == 0) return;

    this->buffer.push_back(this->current_byte);
    this->current_byte = 0;
    this->bit_pos = 0;
}

const std::vector<uint8_t>& BitWriter::get_buffer() const {
    return this->buffer;
}