#pragma once
#include <cstdint>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/bitstream/bit_reader.hpp"

struct FrameHeader {
    uint16_t sync; // 0x4C41 "LA"
    uint8_t version; // 1 for LAC 1.0.0
    uint8_t channels; // 1 or 2
    uint16_t sample_rate; // in hz
    uint8_t bit_depth;
    uint16_t block_size;
    uint8_t reserved; // must be 0

    FrameHeader() : sync(0x4C41), version(1), channels(2), sample_rate(44100), bit_depth(16), block_size(4096), reserved(0) {}

    void write(BitWriter& w) const {
        w.write_bits(this->sync, 16);
        w.write_bits(this->version, 8);
        w.write_bits(this->channels, 8);
        w.write_bits(this->sample_rate, 16);
        w.write_bits(this->bit_depth, 8);
        w.write_bits(this->block_size, 16);
        w.write_bits(this->reserved, 8);
    }

    void read(BitReader& r) {
        this->sync = r.read_bits(16);
        this->version = r.read_bits(8);
        this->channels = r.read_bits(8);
        this->sample_rate = r.read_bits(16);
        this->bit_depth = r.read_bits(8);
        this->block_size = r.read_bits(16);
        this->reserved = r.read_bits(8);
    }
};