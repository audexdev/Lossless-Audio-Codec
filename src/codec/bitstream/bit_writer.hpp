#pragma once
#include <cstdint>
#include <vector>

class BitWriter {
public:
    BitWriter();

    void write_bit(uint32_t bit);
    void write_bits(uint32_t value, int nbits);
    void flush_to_byte();
    const std::vector<uint8_t>& get_buffer() const;

private:
    std::vector<uint8_t> buffer;
    uint8_t current_byte;
    int bit_pos;
};