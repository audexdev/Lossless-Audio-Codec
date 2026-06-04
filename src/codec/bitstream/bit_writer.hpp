#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

class BitWriter {
public:
    BitWriter();

    void write_bit(uint32_t bit);
    void write_bits(uint32_t value, int nbits);
    void write_unary_ones(uint32_t ones);
    void write_bytes(const uint8_t* data, size_t size);
    void reserve_bytes(size_t size);
    void flush_to_byte();
    const std::vector<uint8_t>& get_buffer() const;
    std::vector<uint8_t> take_buffer();

private:
    std::vector<uint8_t> buffer;
    uint8_t current_byte;
    int bit_pos;
};
