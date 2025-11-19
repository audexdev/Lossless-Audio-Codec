#include "wav_io.hpp"
#include <fstream>

static uint16_t read_u16_le(std::ifstream& f) {
    uint8_t b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    return static_cast<uint16_t>(b[0] | (static_cast<uint16_t>(b[1]) << 8));
}

static uint32_t read_u32_le(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0] |
                                 (static_cast<uint32_t>(b[1]) << 8) |
                                 (static_cast<uint32_t>(b[2]) << 16) |
                                 (static_cast<uint32_t>(b[3]) << 24));
}

static void write_u16_le(std::ofstream& f, uint16_t v) {
    uint8_t b[2] = { static_cast<uint8_t>(v & 0xFF), static_cast<uint8_t>((v >> 8) & 0xFF) };
    f.write(reinterpret_cast<const char*>(b), 2);
}

static void write_u32_le(std::ofstream& f, uint32_t v) {
    uint8_t b[4] = {
        static_cast<uint8_t>(v & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF),
        static_cast<uint8_t>((v >> 24) & 0xFF)
    };
    f.write(reinterpret_cast<const char*>(b), 4);
}

static int16_t clip_int16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}

bool read_wav(const std::string& path,
              std::vector<int32_t>& left,
              std::vector<int32_t>& right,
              uint16_t& channels,
              uint32_t& sample_rate) {
    left.clear();
    right.clear();
    channels = 0;
    sample_rate = 0;

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
            if (chunk_size > 16) {
                f.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
            got_fmt = true;
        } else if (std::string(chunk_id, 4) == "data") {
            if (!got_fmt) return false;
            if (fmt_audio_format != 1 || fmt_bits_per_sample != 16) return false;
            if (sample_rate != 44100) return false;
            if (channels != 1 && channels != 2) return false;
            if (fmt_block_align != channels * 2) return false;

            std::vector<char> raw(chunk_size);
            f.read(raw.data(), static_cast<std::streamsize>(chunk_size));
            if (f.gcount() != static_cast<std::streamsize>(chunk_size)) return false;

            size_t samples = chunk_size / (channels * 2);
            left.resize(samples);
            if (channels == 2) right.resize(samples);

            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.data());
            for (size_t i = 0; i < samples; ++i) {
                int16_t l = static_cast<int16_t>(p[0] | (p[1] << 8));
                p += 2;
                left[i] = static_cast<int32_t>(l);
                if (channels == 2) {
                    int16_t r = static_cast<int16_t>(p[0] | (p[1] << 8));
                    p += 2;
                    right[i] = static_cast<int32_t>(r);
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
               uint32_t sample_rate) {
    if (channels != 1 && channels != 2) return false;
    if (sample_rate != 44100) return false;
    if (channels == 2 && left.size() != right.size()) return false;

    uint32_t frames = static_cast<uint32_t>(left.size());
    uint32_t data_size = frames * channels * 2;
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
    write_u32_le(f, sample_rate * channels * 2);
    write_u16_le(f, channels * 2);
    write_u16_le(f, 16);

    f.write("data", 4);
    write_u32_le(f, data_size);

    for (uint32_t i = 0; i < frames; ++i) {
        int16_t l = clip_int16(left[i]);
        write_u16_le(f, static_cast<uint16_t>(l));
        if (channels == 2) {
            int16_t r = clip_int16(right[i]);
            write_u16_le(f, static_cast<uint16_t>(r));
        }
    }

    return f.good();
}
