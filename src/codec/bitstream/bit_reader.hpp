#pragma once
#include <cstdint>
#include <vector>

class BitReader {
public:
    BitReader(const uint8_t* data, size_t size);
    BitReader(const std::vector<uint8_t>& buf);

    uint32_t read_bit();
    uint32_t read_bits(int nbits);

    void align_to_byte();
    bool eof() const;
    bool has_error() const;
    size_t bits_remaining() const;

private:
    const uint8_t* data;
    size_t size;
    size_t byte_pos;
    int bit_pos;
    bool error;

    void mark_error();
};
