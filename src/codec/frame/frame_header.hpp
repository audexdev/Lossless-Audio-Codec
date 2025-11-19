#pragma once
#include <cstdint>
#include <cstddef>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/bitstream/bit_reader.hpp"

struct FrameHeader {
    uint16_t sync; // 0x4C41 "LA"
    uint8_t version; // 1 for LAC 1.0.0
    uint8_t channels; // 1 or 2
    uint8_t stereo_mode; // 0=LR,1=MS (stereo only)
    uint16_t sample_rate; // in hz
    uint8_t bit_depth;
    uint16_t block_size;
    uint8_t reserved; // must be 0

    FrameHeader() : sync(0x4C41), version(1), channels(2), stereo_mode(0), sample_rate(44100), bit_depth(16), block_size(4096), reserved(0) {}

    void write(BitWriter& w) const {
        w.write_bits(this->sync, 16);
        w.write_bits(this->version, 8);
        w.write_bits(this->channels, 8);
        w.write_bits(this->stereo_mode, 8);
        w.write_bits(this->sample_rate, 16);
        w.write_bits(this->bit_depth, 8);
        w.write_bits(this->block_size, 16);
        w.write_bits(this->reserved, 8);
    }

    void read(BitReader& r) {
        this->sync = r.read_bits(16);
        this->version = r.read_bits(8);
        this->channels = r.read_bits(8);
        this->stereo_mode = r.read_bits(8);
        this->sample_rate = r.read_bits(16);
        this->bit_depth = r.read_bits(8);
        this->block_size = r.read_bits(16);
        this->reserved = r.read_bits(8);
    }

    bool validate() const {
        if (this->sync != 0x4C41 || this->version != 1) return false;
        if (this->channels != 1 && this->channels != 2) return false;
        if (this->channels == 1 && this->stereo_mode != 0) return false;
        if (this->stereo_mode != 0 && this->stereo_mode != 1) return false;
        if (this->sample_rate != 44100 || this->bit_depth != 16) return false;
        if (this->block_size == 0 || this->reserved != 0) return false;
        return true;
    }

    static bool parse(const uint8_t* data, size_t size, FrameHeader& hdr, size_t& header_bytes) {
        if (size < 10) return false;

        if (size >= 11) {
            BitReader br(data, size);
            hdr.read(br);
            if (hdr.validate()) {
                header_bytes = 11;
                return true;
            }
        }

        BitReader br_legacy(data, size);
        hdr.sync = br_legacy.read_bits(16);
        hdr.version = br_legacy.read_bits(8);
        hdr.channels = br_legacy.read_bits(8);
        hdr.stereo_mode = 0;
        hdr.sample_rate = br_legacy.read_bits(16);
        hdr.bit_depth = br_legacy.read_bits(8);
        hdr.block_size = br_legacy.read_bits(16);
        hdr.reserved = br_legacy.read_bits(8);
        if (hdr.validate()) {
            header_bytes = 10;
            return true;
        }

        return false;
    }
};
