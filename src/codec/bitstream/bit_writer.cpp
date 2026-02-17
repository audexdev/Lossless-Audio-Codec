#include "bit_writer.hpp"
#include <algorithm>

namespace {
inline uint32_t low_bits_mask(int bits) {
    if (bits <= 0) return 0u;
    if (bits >= 32) return 0xFFFFFFFFu;
    return (1u << bits) - 1u;
}
} // namespace

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
    if (nbits <= 0) return;

    // Keep legacy behavior for uncommon widths outside uint32 payload size.
    if (nbits > 32) {
        for (int i = nbits - 1; i >= 0; --i) {
            const uint32_t bit = (i >= 32) ? 0u : ((value >> i) & 1u);
            this->write_bit(bit);
        }
        return;
    }

    uint64_t acc = static_cast<uint64_t>(this->current_byte) << 56;
    int acc_bits = this->bit_pos;

    auto flush_full_bytes = [&]() {
        while (acc_bits >= 8) {
            this->buffer.push_back(static_cast<uint8_t>(acc >> 56));
            acc <<= 8;
            acc_bits -= 8;
        }
    };

    while (nbits > 0) {
        const int take = std::min(nbits, 32 - acc_bits);
        const int src_shift = nbits - take;
        const uint32_t chunk = (value >> src_shift) & low_bits_mask(take);

        acc |= static_cast<uint64_t>(chunk) << (64 - acc_bits - take);
        acc_bits += take;
        nbits -= take;
        flush_full_bytes();
    }

    if (acc_bits > 0) {
        this->current_byte = static_cast<uint8_t>(acc >> 56);
        this->bit_pos = acc_bits;
    } else {
        this->current_byte = 0;
        this->bit_pos = 0;
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
