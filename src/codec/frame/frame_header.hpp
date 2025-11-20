#pragma once
#include <cstdint>
#include <cstddef>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/bitstream/bit_reader.hpp"

struct FrameHeader {
    uint16_t sync; // 0x4C41 "LA"
    uint8_t version; // 2 for adaptive block format
    uint8_t channels; // 1 or 2
    uint8_t stereo_mode; // 0=LR,1=MS,2=per-block (stereo only)
    uint32_t sample_rate; // in hz
    uint8_t bit_depth;
    uint8_t reserved;

    FrameHeader()
        : sync(0x4C41),
          version(2),
          channels(2),
          stereo_mode(2),
          sample_rate(44100),
          bit_depth(16),
          reserved(0) {}

    void write(BitWriter& w) const {
        w.write_bits(this->sync, 16);
        w.write_bits(this->version, 8);
        w.write_bits(this->channels, 8);
        w.write_bits(this->stereo_mode, 8);
        uint16_t sample_rate_low = static_cast<uint16_t>(this->sample_rate & 0xFFFF);
        uint8_t sample_rate_high = static_cast<uint8_t>((this->sample_rate >> 16) & 0xFF);
        w.write_bits(sample_rate_low, 16);
        w.write_bits(sample_rate_high, 8);
        w.write_bits(this->bit_depth, 8);
        w.write_bits(this->reserved, 8);
    }

    void read(BitReader& r) {
        this->sync = r.read_bits(16);
        this->version = r.read_bits(8);
        this->channels = r.read_bits(8);
        this->stereo_mode = r.read_bits(8);
        uint16_t sample_rate_low = r.read_bits(16);
        uint8_t sample_rate_high = r.read_bits(8);
        this->bit_depth = r.read_bits(8);
        this->reserved = r.read_bits(8);
        this->sample_rate = sample_rate_low | (static_cast<uint32_t>(sample_rate_high) << 16);
    }

    bool validate() const {
        if (this->sync != 0x4C41 || this->version != 2) return false;
        if (this->channels != 1 && this->channels != 2) return false;
        if (this->channels == 1 && this->stereo_mode != 0) return false;
        if (this->stereo_mode != 0 && this->stereo_mode != 1 && this->stereo_mode != 2) return false;
        if (!is_supported_sample_rate(this->sample_rate)) return false;
        if (this->bit_depth != 16 && this->bit_depth != 24) return false;
        return true;
    }

    static bool parse(const uint8_t* data, size_t size, FrameHeader& hdr, size_t& header_bytes) {
        if (size < 10) return false;

        BitReader br(data, size);
        hdr.read(br);
        if (!br.has_error() && hdr.validate()) {
            header_bytes = 10;
            return true;
        }

        return false;
    }

private:
    static bool is_supported_sample_rate(uint32_t sr) {
        return sr == 44100 || sr == 48000 || sr == 96000 || sr == 192000;
    }
};
