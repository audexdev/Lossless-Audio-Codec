#include "wav_io.hpp"
#include <fstream>
#include <limits>

namespace {

constexpr int32_t PCM16_MIN = -32768;
constexpr int32_t PCM16_MAX = 32767;
constexpr int32_t PCM24_MIN = -0x800000;
constexpr int32_t PCM24_MAX = 0x7FFFFF;
constexpr uint64_t MAX_DECODED_PCM_BYTES = 1ULL << 30;

bool is_supported_sample_rate(uint32_t sr) {
    return sr == 44100 || sr == 48000 || sr == 96000 || sr == 192000;
}

bool is_supported_bit_depth(uint16_t bits) {
    return bits == 16 || bits == 24;
}

bool read_exact(std::ifstream& f, char* data, std::streamsize size) {
    f.read(data, size);
    return f.gcount() == size;
}

bool read_u16_le(std::ifstream& f, uint16_t& value) {
    uint8_t b[2];
    if (!read_exact(f, reinterpret_cast<char*>(b), 2)) return false;
    value = static_cast<uint16_t>(b[0] | (static_cast<uint16_t>(b[1]) << 8));
    return true;
}

bool read_u32_le(std::ifstream& f, uint32_t& value) {
    uint8_t b[4];
    if (!read_exact(f, reinterpret_cast<char*>(b), 4)) return false;
    value = static_cast<uint32_t>(b[0] |
                                  (static_cast<uint32_t>(b[1]) << 8) |
                                  (static_cast<uint32_t>(b[2]) << 16) |
                                  (static_cast<uint32_t>(b[3]) << 24));
    return true;
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

bool is_valid_sample_for_depth(int32_t sample, uint8_t bit_depth) {
    if (bit_depth == 16) return sample >= PCM16_MIN && sample <= PCM16_MAX;
    if (bit_depth == 24) return sample >= PCM24_MIN && sample <= PCM24_MAX;
    return false;
}

bool seek_forward(std::ifstream& f, uint64_t bytes) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
    f.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    return f.good();
}

bool read_pcm16_sample(std::ifstream& f, int32_t& sample) {
    uint8_t b[2];
    if (!read_exact(f, reinterpret_cast<char*>(b), 2)) return false;
    sample = static_cast<int32_t>(static_cast<uint32_t>(b[0]) |
                                  (static_cast<uint32_t>(b[1]) << 8));
    if (sample & 0x8000) sample |= ~0xFFFF;
    return true;
}

bool read_pcm24_sample(std::ifstream& f, int32_t& sample) {
    uint8_t b[3];
    if (!read_exact(f, reinterpret_cast<char*>(b), 3)) return false;
    sample = static_cast<int32_t>(static_cast<uint32_t>(b[0]) |
                                  (static_cast<uint32_t>(b[1]) << 8) |
                                  (static_cast<uint32_t>(b[2]) << 16));
    if (sample & 0x800000) sample |= ~0xFFFFFF;
    return true;
}

void write_pcm16_sample(std::ofstream& f, int32_t v) {
    write_u16_le(f, static_cast<uint16_t>(v));
}

void write_pcm24_sample(std::ofstream& f, int32_t v) {
    uint8_t b[3] = {
        static_cast<uint8_t>(v & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF)
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

    f.seekg(0, std::ios::end);
    const std::streampos end = f.tellg();
    if (end < 12) return false;
    const uint64_t file_size = static_cast<uint64_t>(end);
    f.seekg(0, std::ios::beg);

    char riff[4];
    if (!read_exact(f, riff, 4) || std::string(riff, 4) != "RIFF") return false;
    uint32_t riff_size = 0;
    if (!read_u32_le(f, riff_size) || static_cast<uint64_t>(riff_size) + 8u != file_size) return false;
    char wave[4];
    if (!read_exact(f, wave, 4) || std::string(wave, 4) != "WAVE") return false;

    bool got_fmt = false;
    bool got_data = false;
    uint16_t fmt_audio_format = 0;
    uint16_t fmt_bits_per_sample = 0;
    uint16_t fmt_block_align = 0;
    uint16_t parsed_channels = 0;
    uint32_t parsed_sample_rate = 0;
    uint8_t parsed_bit_depth = 0;
    std::vector<int32_t> parsed_left;
    std::vector<int32_t> parsed_right;
    uint64_t remaining = file_size - 12u;

    while (remaining > 0) {
        if (remaining < 8u) return false;
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (!read_exact(f, chunk_id, 4) || !read_u32_le(f, chunk_size)) return false;
        remaining -= 8u;
        const uint64_t padded_chunk_size = static_cast<uint64_t>(chunk_size) + (chunk_size & 1u);
        if (padded_chunk_size > remaining) return false;

        if (std::string(chunk_id, 4) == "fmt ") {
            if (got_fmt || got_data || chunk_size != 16u) return false;
            uint32_t byte_rate = 0;
            if (!read_u16_le(f, fmt_audio_format) ||
                !read_u16_le(f, parsed_channels) ||
                !read_u32_le(f, parsed_sample_rate) ||
                !read_u32_le(f, byte_rate) ||
                !read_u16_le(f, fmt_block_align) ||
                !read_u16_le(f, fmt_bits_per_sample)) {
                return false;
            }
            if (fmt_audio_format != 1) return false;
            if (!is_supported_bit_depth(fmt_bits_per_sample)) return false;
            if (!is_supported_sample_rate(parsed_sample_rate)) return false;
            if (parsed_channels != 1 && parsed_channels != 2) return false;
            const uint16_t bytes_per_sample = static_cast<uint16_t>(fmt_bits_per_sample / 8);
            const uint16_t expected_block_align = static_cast<uint16_t>(parsed_channels * bytes_per_sample);
            if (fmt_block_align != expected_block_align) return false;
            if (byte_rate != parsed_sample_rate * expected_block_align) return false;
            parsed_bit_depth = static_cast<uint8_t>(fmt_bits_per_sample);
            got_fmt = true;
        } else if (std::string(chunk_id, 4) == "data") {
            if (!got_fmt || got_data || chunk_size == 0u) return false;
            uint16_t bytes_per_sample = static_cast<uint16_t>(fmt_bits_per_sample / 8);
            if (chunk_size % fmt_block_align != 0) return false;

            const size_t samples = chunk_size / fmt_block_align;
            const uint64_t decoded_pcm_bytes =
                static_cast<uint64_t>(samples) * parsed_channels * sizeof(int32_t);
            if (decoded_pcm_bytes > MAX_DECODED_PCM_BYTES) return false;
            parsed_left.resize(samples);
            if (parsed_channels == 2) {
                parsed_right.resize(samples);
            } else {
                parsed_right.clear();
            }

            for (size_t i = 0; i < samples; ++i) {
                if (fmt_bits_per_sample == 16) {
                    if (!read_pcm16_sample(f, parsed_left[i])) return false;
                } else {
                    if (!read_pcm24_sample(f, parsed_left[i])) return false;
                }
                if (parsed_channels == 2) {
                    if (fmt_bits_per_sample == 16) {
                        if (!read_pcm16_sample(f, parsed_right[i])) return false;
                    } else {
                        if (!read_pcm24_sample(f, parsed_right[i])) return false;
                    }
                }
            }
            got_data = true;
        } else {
            if (!seek_forward(f, chunk_size)) return false;
        }

        if ((chunk_size & 1u) != 0u && !seek_forward(f, 1u)) return false;
        remaining -= padded_chunk_size;
    }

    if (!got_fmt || !got_data) return false;
    left.swap(parsed_left);
    right.swap(parsed_right);
    channels = parsed_channels;
    sample_rate = parsed_sample_rate;
    bit_depth = parsed_bit_depth;
    return true;
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
    if (left.empty()) return false;
    if (channels == 1 && !right.empty()) return false;
    if (channels == 2 && left.size() != right.size()) return false;
    for (int32_t sample : left) {
        if (!is_valid_sample_for_depth(sample, bit_depth)) return false;
    }
    for (int32_t sample : right) {
        if (!is_valid_sample_for_depth(sample, bit_depth)) return false;
    }

    uint16_t bytes_per_sample = static_cast<uint16_t>(bit_depth / 8);
    uint32_t block_align = channels * bytes_per_sample;
    const uint64_t data_size_u64 = static_cast<uint64_t>(left.size()) * block_align;
    const uint64_t data_padding_u64 = data_size_u64 & 1u;
    const uint64_t riff_size_u64 = 36u + data_size_u64 + data_padding_u64;
    if (riff_size_u64 > std::numeric_limits<uint32_t>::max()) return false;
    const uint32_t frames = static_cast<uint32_t>(left.size());
    const uint32_t data_size = static_cast<uint32_t>(data_size_u64);
    const uint32_t riff_size = static_cast<uint32_t>(riff_size_u64);

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

    if (data_padding_u64 != 0u) {
        const char padding = 0;
        f.write(&padding, 1);
    }
    f.close();
    return f.good();
}
