#include "wav_io.hpp"
#include <fstream>

namespace {

constexpr int32_t PCM16_MIN = -32768;
constexpr int32_t PCM16_MAX = 32767;
constexpr int32_t PCM24_MIN = -0x800000;
constexpr int32_t PCM24_MAX = 0x7FFFFF;

bool is_supported_sample_rate(uint32_t sr) {
    return sr == 44100 || sr == 48000 || sr == 96000 || sr == 192000;
}

bool is_supported_bit_depth(uint16_t bits) {
    return bits == 16 || bits == 24;
}

uint16_t read_u16_le(std::ifstream& f) {
    uint8_t b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    return static_cast<uint16_t>(b[0] | (static_cast<uint16_t>(b[1]) << 8));
}

uint32_t read_u32_le(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0] |
                                 (static_cast<uint32_t>(b[1]) << 8) |
                                 (static_cast<uint32_t>(b[2]) << 16) |
                                 (static_cast<uint32_t>(b[3]) << 24));
}

void write_u16_le(std::ofstream& f, uint16_t v) {
    uint8_t b[2] = { static_cast<uint8_t>(v & 0xFF), static_cast<uint8_t>((v >> 8) & 0xFF) };
    f.write(reinterpret_cast<const char*>(b), 2);
}

void write_u32_le(std::ofstream& f, uint32_t v) {
    uint8_t b[4] = {
        static_cast<uint8_t>(v & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF),
        static_cast<uint8_t>((v >> 24) & 0xFF)
    };
    f.write(reinterpret_cast<const char*>(b), 4);
}

int16_t clip_int16(int32_t v) {
    if (v > PCM16_MAX) return PCM16_MAX;
    if (v < PCM16_MIN) return PCM16_MIN;
    return static_cast<int16_t>(v);
}

int32_t clip_int24(int32_t v) {
    if (v > PCM24_MAX) return PCM24_MAX;
    if (v < PCM24_MIN) return PCM24_MIN;
    return v;
}

int32_t read_pcm16_sample(const uint8_t*& p) {
    int32_t sample = static_cast<int32_t>(static_cast<uint32_t>(p[0]) |
                                          (static_cast<uint32_t>(p[1]) << 8));
    p += 2;
    if (sample & 0x8000) sample |= ~0xFFFF;
    return sample;
}

int32_t read_pcm24_sample(const uint8_t*& p) {
    int32_t sample = static_cast<int32_t>(static_cast<uint32_t>(p[0]) |
                                          (static_cast<uint32_t>(p[1]) << 8) |
                                          (static_cast<uint32_t>(p[2]) << 16));
    p += 3;
    if (sample & 0x800000) sample |= ~0xFFFFFF;
    return sample;
}

void write_pcm16_sample(std::ofstream& f, int32_t v) {
    int16_t clipped = clip_int16(v);
    write_u16_le(f, static_cast<uint16_t>(clipped));
}

void write_pcm24_sample(std::ofstream& f, int32_t v) {
    int32_t sample = clip_int24(v);
    uint8_t b[3] = {
        static_cast<uint8_t>(sample & 0xFF),
        static_cast<uint8_t>((sample >> 8) & 0xFF),
        static_cast<uint8_t>((sample >> 16) & 0xFF)
    };
    f.write(reinterpret_cast<const char*>(b), 3);
}

} // namespace

bool read_wav(const std::string& path,
              std::vector<int32_t>& left,
              std::vector<int32_t>& right,
              uint16_t& channels,
              uint32_t& sample_rate,
              uint8_t& bit_depth) {
    left.clear();
    right.clear();
    channels = 0;
    sample_rate = 0;
    bit_depth = 0;

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char riff[4];
    f.read(riff, 4);
    if (f.gcount() != 4 || std::string(riff, 4) != "RIFF") return false;
    uint32_t riff_size = read_u32_le(f);
    (void)riff_size;
    char wave[4];
    f.read(wave, 4);
    if (f.gcount() != 4 || std::string(wave, 4) != "WAVE") return false;

    bool got_fmt = false;
    bool got_data = false;
    uint16_t fmt_audio_format = 0;
    uint16_t fmt_bits_per_sample = 0;
    uint16_t fmt_block_align = 0;

    while (f && !got_data) {
        char chunk_id[4];
        f.read(chunk_id, 4);
        if (f.gcount() != 4) break;
        uint32_t chunk_size = read_u32_le(f);

        if (std::string(chunk_id, 4) == "fmt ") {
            fmt_audio_format = read_u16_le(f);
            channels = read_u16_le(f);
            sample_rate = read_u32_le(f);
            uint32_t byte_rate = read_u32_le(f);
            (void)byte_rate;
            fmt_block_align = read_u16_le(f);
            fmt_bits_per_sample = read_u16_le(f);
            bit_depth = static_cast<uint8_t>(fmt_bits_per_sample);
            if (chunk_size > 16) {
                f.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
            got_fmt = true;
        } else if (std::string(chunk_id, 4) == "data") {
            if (!got_fmt) return false;
            if (fmt_audio_format != 1) return false;
            if (!is_supported_bit_depth(fmt_bits_per_sample)) return false;
            if (!is_supported_sample_rate(sample_rate)) return false;
            if (channels != 1 && channels != 2) return false;
            uint16_t bytes_per_sample = static_cast<uint16_t>(fmt_bits_per_sample / 8);
            if (fmt_block_align != channels * bytes_per_sample) return false;

            if (chunk_size % (channels * bytes_per_sample) != 0) return false;
            std::vector<char> raw(chunk_size);
            f.read(raw.data(), static_cast<std::streamsize>(chunk_size));
            if (f.gcount() != static_cast<std::streamsize>(chunk_size)) return false;

            size_t samples = chunk_size / (channels * bytes_per_sample);
            left.resize(samples);
            if (channels == 2) {
                right.resize(samples);
            } else {
                right.clear();
            }

            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.data());
            for (size_t i = 0; i < samples; ++i) {
                if (fmt_bits_per_sample == 16) {
                    left[i] = read_pcm16_sample(p);
                } else {
                    left[i] = read_pcm24_sample(p);
                }
                if (channels == 2) {
                    if (fmt_bits_per_sample == 16) {
                        right[i] = read_pcm16_sample(p);
                    } else {
                        right[i] = read_pcm24_sample(p);
                    }
                }
            }
            got_data = true;
        } else {
            f.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }

        if (chunk_size % 2 == 1 && f) f.seekg(1, std::ios::cur);
    }

    return got_fmt && got_data;
}

bool write_wav(const std::string& path,
               const std::vector<int32_t>& left,
               const std::vector<int32_t>& right,
               uint16_t channels,
               uint32_t sample_rate,
               uint8_t bit_depth) {
    if (channels != 1 && channels != 2) return false;
    if (!is_supported_sample_rate(sample_rate)) return false;
    if (bit_depth != 16 && bit_depth != 24) return false;
    if (channels == 2 && left.size() != right.size()) return false;

    uint32_t frames = static_cast<uint32_t>(left.size());
    uint16_t bytes_per_sample = static_cast<uint16_t>(bit_depth / 8);
    uint32_t block_align = channels * bytes_per_sample;
    uint32_t data_size = frames * block_align;
    uint32_t riff_size = 36 + data_size;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    f.write("RIFF", 4);
    write_u32_le(f, riff_size);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    write_u32_le(f, 16);
    write_u16_le(f, 1);
    write_u16_le(f, channels);
    write_u32_le(f, sample_rate);
    write_u32_le(f, sample_rate * block_align);
    write_u16_le(f, static_cast<uint16_t>(block_align));
    write_u16_le(f, bit_depth);

    f.write("data", 4);
    write_u32_le(f, data_size);

    for (uint32_t i = 0; i < frames; ++i) {
        if (bit_depth == 16) {
            write_pcm16_sample(f, left[i]);
        } else {
            write_pcm24_sample(f, left[i]);
        }

        if (channels == 2) {
            if (bit_depth == 16) {
                write_pcm16_sample(f, right[i]);
            } else {
                write_pcm24_sample(f, right[i]);
            }
        }
    }

    return f.good();
}
