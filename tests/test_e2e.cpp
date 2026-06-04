#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"
#include "codec/block/constants.hpp"
#include "codec/block/encoder.hpp"
#include "io/wav_io.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/rice/rice.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::filesystem::path assets_dir() {
#ifdef LAC_ASSETS_DIR
    return std::filesystem::path(LAC_ASSETS_DIR);
#else
    return std::filesystem::current_path() / "../assets";
#endif
}

struct WavData {
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint8_t bit_depth = 0;
};

bool load_wav_file(const std::filesystem::path& path, WavData& out) {
    out.left.clear();
    out.right.clear();
    return read_wav(path.string(), out.left, out.right, out.channels, out.sample_rate, out.bit_depth);
}

std::vector<uint8_t> load_binary_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f.good() && "Failed to open binary file");

    f.seekg(0, std::ios::end);
    const std::streampos end = f.tellg();
    assert(end >= 0 && "Failed to determine binary file size");

    std::vector<uint8_t> data(static_cast<size_t>(end));
    f.seekg(0, std::ios::beg);
    if (!data.empty()) {
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        assert(f.good() && "Failed to read binary file");
    }
    return data;
}

std::filesystem::path write_temp_path(const std::string& ext) {
    static uint64_t counter = 0;
    auto base = std::filesystem::temp_directory_path() / ("lac_test_" + std::to_string(++counter) + ext);
    return base;
}

struct CompareResult {
    bool equal = true;
    size_t mismatches = 0;
    int32_t max_abs_diff = 0;
};

CompareResult compare_pcm(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    CompareResult res;
    if (a.size() != b.size()) {
        res.equal = false;
        res.mismatches = (a.size() > b.size()) ? (a.size() - b.size()) : (b.size() - a.size());
        return res;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++res.mismatches;
            int64_t diff = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[i]);
            int32_t adiff = static_cast<int32_t>(diff >= 0 ? diff : -diff);
            if (adiff > res.max_abs_diff) res.max_abs_diff = adiff;
        }
    }
    res.equal = (res.mismatches == 0);
    return res;
}

void run_case_path(const std::filesystem::path& src_path,
                   const std::string& label,
                   uint8_t requested_stereo_mode = 2) {
    std::cout << "[e2e] start " << label << "\n";
    WavData orig;
    bool ok = load_wav_file(src_path, orig);
    assert(ok && "Failed to load source WAV");

    uint8_t stereo_mode = (orig.channels == 2) ? requested_stereo_mode : 0;
    LAC::Encoder encoder(12, stereo_mode, orig.sample_rate, orig.bit_depth, false, false, false);
    std::vector<uint8_t> bitstream = encoder.encode(orig.left, orig.right);

    const auto lac_path = write_temp_path(".lac");
    {
        std::ofstream f(lac_path, std::ios::binary);
        assert(f.good());
        f.write(reinterpret_cast<const char*>(bitstream.data()), static_cast<std::streamsize>(bitstream.size()));
        assert(f.good() && "Failed to write temporary LAC file");
    }

    std::vector<uint8_t> encoded_file = load_binary_file(lac_path);
    assert(encoded_file == bitstream && "Temporary LAC file bytes differ from encoder output");

    LAC::Decoder decoder;
    std::vector<int32_t> dec_left, dec_right;
    FrameHeader hdr;
    try {
        decoder.decode(encoded_file.data(), encoded_file.size(), dec_left, dec_right, &hdr);
    } catch (const std::runtime_error& e) {
        std::cerr << "[e2e-decode-error] file=" << label << " error=" << e.what() << "\n";
        assert(false && "Decode failed");
    }

    const auto out_wav_path = write_temp_path(".wav");
    ok = write_wav(out_wav_path.string(), dec_left, dec_right, hdr.channels, hdr.sample_rate, hdr.bit_depth);
    assert(ok && "Write restored WAV failed");

    WavData restored;
    ok = load_wav_file(out_wav_path, restored);
    assert(ok && "Reload restored WAV failed");

    assert(restored.channels == orig.channels);
    assert(restored.sample_rate == orig.sample_rate);
    assert(restored.bit_depth == orig.bit_depth);
    assert(restored.left.size() == orig.left.size());
    assert(restored.right.size() == orig.right.size());

    CompareResult left_cmp = compare_pcm(orig.left, restored.left);
    CompareResult right_cmp = compare_pcm(orig.right, restored.right);
    if (!left_cmp.equal || !right_cmp.equal) {
        std::cerr << "[e2e-mismatch] file=" << label
                  << " left_mis=" << left_cmp.mismatches << " left_maxdiff=" << left_cmp.max_abs_diff
                  << " right_mis=" << right_cmp.mismatches << " right_maxdiff=" << right_cmp.max_abs_diff
                  << "\n";
        assert(false && "PCM mismatch");
    }

    std::error_code ec;
    std::filesystem::remove(lac_path, ec);
    std::filesystem::remove(out_wav_path, ec);
    std::cout << "[e2e] ok " << label << "\n";
}

void run_case(const std::string& filename) {
    run_case_path(assets_dir() / filename, filename);
}

std::filesystem::path write_generated_wav(const std::string& label,
                                          uint16_t channels,
                                          uint32_t sample_rate,
                                          uint8_t bit_depth,
                                          size_t frames) {
    std::vector<int32_t> left(frames);
    std::vector<int32_t> right;
    if (channels == 2) right.resize(frames);

    const int32_t amplitude = (bit_depth == 24) ? 0x300000 : 12000;
    const int32_t min_sample = (bit_depth == 24) ? -0x800000 : -32768;
    const int32_t max_sample = (bit_depth == 24) ? 0x7FFFFF : 32767;
    for (size_t i = 0; i < frames; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(sample_rate);
        left[i] = static_cast<int32_t>(std::sin(2.0 * 3.14159265358979323846 * 440.0 * t) * amplitude);
        if (channels == 2) {
            right[i] = static_cast<int32_t>(std::sin(2.0 * 3.14159265358979323846 * 661.0 * t) * (amplitude / 2));
        }
    }
    if (frames >= 16) {
        left[0] = min_sample;
        left[1] = max_sample;
        for (size_t i = 2; i < 16; ++i) left[i] = 0;
        if (channels == 2) {
            right[0] = max_sample;
            right[1] = min_sample;
            for (size_t i = 2; i < 16; ++i) right[i] = 0;
        }
    }

    const auto wav_path = write_temp_path("_" + label + ".wav");
    const bool ok = write_wav(wav_path.string(), left, right, channels, sample_rate, bit_depth);
    assert(ok && "Failed to write generated WAV fixture");
    return wav_path;
}

void run_generated_cases() {
    std::vector<std::filesystem::path> generated_paths;
    for (uint16_t channels : {1u, 2u}) {
        for (uint32_t sample_rate : {44100u, 48000u, 96000u, 192000u}) {
            for (uint8_t bit_depth : {16u, 24u}) {
                const std::string label =
                    "generated_" + std::to_string(channels) + "ch_" +
                    std::to_string(bit_depth) + "_" + std::to_string(sample_rate);
                const auto path = write_generated_wav(label, channels, sample_rate, bit_depth, 1024);
                generated_paths.push_back(path);
                run_case_path(path, label + ".wav");
                if (channels == 2 && sample_rate == 48000 && bit_depth == 24) {
                    run_case_path(path, label + "_lr.wav", 0);
                    run_case_path(path, label + "_ms.wav", 1);
                }
            }
        }
    }
    const auto multiblock_path =
        write_generated_wav("generated_multiblock", 2, 44100, 16, Block::MAX_BLOCK_SIZE + 37u);
    generated_paths.push_back(multiblock_path);
    run_case_path(multiblock_path, "generated_multiblock.wav");
    std::error_code ec;
    for (const auto& path : generated_paths) {
        std::filesystem::remove(path, ec);
    }
}

} // namespace

namespace {

void append_u16_le(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
}

void append_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
}

void set_u32_le(std::vector<uint8_t>& out, size_t offset, uint32_t value) {
    assert(offset + 4 <= out.size());
    out[offset] = static_cast<uint8_t>(value & 0xFFu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    out[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    out[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

void set_u16_le(std::vector<uint8_t>& out, size_t offset, uint16_t value) {
    assert(offset + 2 <= out.size());
    out[offset] = static_cast<uint8_t>(value & 0xFFu);
    out[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
}

uint32_t get_u32_le(const std::vector<uint8_t>& in, size_t offset) {
    assert(offset + 4 <= in.size());
    return static_cast<uint32_t>(
        static_cast<uint32_t>(in[offset]) |
        (static_cast<uint32_t>(in[offset + 1]) << 8) |
        (static_cast<uint32_t>(in[offset + 2]) << 16) |
        (static_cast<uint32_t>(in[offset + 3]) << 24));
}

uint32_t get_u32_be(const std::vector<uint8_t>& in, size_t offset) {
    assert(offset + 4 <= in.size());
    return static_cast<uint32_t>(
        (static_cast<uint32_t>(in[offset]) << 24) |
        (static_cast<uint32_t>(in[offset + 1]) << 16) |
        (static_cast<uint32_t>(in[offset + 2]) << 8) |
        static_cast<uint32_t>(in[offset + 3]));
}

void set_u32_be(std::vector<uint8_t>& out, size_t offset, uint32_t value) {
    assert(offset + 4 <= out.size());
    out[offset] = static_cast<uint8_t>((value >> 24) & 0xFFu);
    out[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    out[offset + 2] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    out[offset + 3] = static_cast<uint8_t>(value & 0xFFu);
}

size_t v3_payload_offset(const std::vector<uint8_t>& stream) {
    assert(stream.size() >= 14);
    assert(stream[2] == 3u);
    const uint32_t block_count = get_u32_be(stream, 10);
    const size_t offset = 14u + static_cast<size_t>(block_count) * 8u;
    assert(offset <= stream.size());
    return offset;
}

std::vector<uint32_t> v3_block_sizes(const std::vector<uint8_t>& stream) {
    assert(stream.size() >= 14);
    assert(stream[2] == 3u);
    const uint32_t block_count = get_u32_be(stream, 10);
    assert(14u + static_cast<size_t>(block_count) * 8u <= stream.size());
    std::vector<uint32_t> sizes;
    sizes.reserve(block_count);
    for (uint32_t i = 0; i < block_count; ++i) {
        sizes.push_back(get_u32_be(stream, 14u + static_cast<size_t>(i) * 8u));
    }
    return sizes;
}

std::vector<uint32_t> v3_payload_sizes(const std::vector<uint8_t>& stream) {
    assert(stream.size() >= 14);
    assert(stream[2] == 3u);
    const uint32_t block_count = get_u32_be(stream, 10);
    assert(14u + static_cast<size_t>(block_count) * 8u <= stream.size());
    std::vector<uint32_t> sizes;
    sizes.reserve(block_count);
    for (uint32_t i = 0; i < block_count; ++i) {
        sizes.push_back(get_u32_be(stream, 18u + static_cast<size_t>(i) * 8u));
    }
    return sizes;
}

std::vector<uint8_t> v3_stereo_flags(const std::vector<uint8_t>& stream) {
    assert(stream.size() >= 14);
    assert(stream[2] == 3u);
    assert(stream[4] == 2u);
    const uint32_t block_count = get_u32_be(stream, 10);
    size_t payload_offset = v3_payload_offset(stream);
    std::vector<uint8_t> flags;
    flags.reserve(block_count);
    for (uint32_t i = 0; i < block_count; ++i) {
        const uint32_t payload_size = get_u32_be(stream, 18u + static_cast<size_t>(i) * 8u);
        assert(payload_size > 0u);
        assert(payload_offset + payload_size <= stream.size());
        flags.push_back(stream[payload_offset]);
        payload_offset += payload_size;
    }
    assert(payload_offset == stream.size());
    return flags;
}

void assert_per_block_stereo_payload_consistency(const std::vector<uint8_t>& auto_stream,
                                                 const std::vector<uint8_t>& lr_stream,
                                                 const std::vector<uint8_t>& ms_stream) {
    assert(v3_block_sizes(auto_stream) == v3_block_sizes(lr_stream));
    assert(v3_block_sizes(auto_stream) == v3_block_sizes(ms_stream));

    const std::vector<uint8_t> flags = v3_stereo_flags(auto_stream);
    const std::vector<uint32_t> auto_payloads = v3_payload_sizes(auto_stream);
    const std::vector<uint32_t> lr_payloads = v3_payload_sizes(lr_stream);
    const std::vector<uint32_t> ms_payloads = v3_payload_sizes(ms_stream);
    assert(flags.size() == auto_payloads.size());
    assert(flags.size() == lr_payloads.size());
    assert(flags.size() == ms_payloads.size());

    uint64_t expected_payload_bytes = 0;
    for (size_t i = 0; i < flags.size(); ++i) {
        assert(flags[i] == 0u || flags[i] == 1u);
        const uint32_t selected_payload = (flags[i] == 1u) ? ms_payloads[i] : lr_payloads[i];
        assert(auto_payloads[i] == selected_payload + 1u);
        expected_payload_bytes += auto_payloads[i];
    }
    assert(auto_stream.size() == v3_payload_offset(auto_stream) + expected_payload_bytes);
}

void append_fourcc(std::vector<uint8_t>& out, const char* fourcc) {
    out.insert(out.end(), fourcc, fourcc + 4);
}

void append_chunk(std::vector<uint8_t>& out, const char* id, const std::vector<uint8_t>& payload) {
    append_fourcc(out, id);
    append_u32_le(out, static_cast<uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    if ((payload.size() & 1u) != 0u) out.push_back(0);
}

std::vector<uint8_t> make_pcm16_mono_wav(bool include_odd_junk = false,
                                         size_t fmt_size = 16,
                                         size_t data_size = 2) {
    std::vector<uint8_t> out;
    append_fourcc(out, "RIFF");
    append_u32_le(out, 0);
    append_fourcc(out, "WAVE");
    if (include_odd_junk) append_chunk(out, "JUNK", {'x'});

    std::vector<uint8_t> fmt;
    append_u16_le(fmt, 1);
    append_u16_le(fmt, 1);
    append_u32_le(fmt, 44100);
    append_u32_le(fmt, 88200);
    append_u16_le(fmt, 2);
    append_u16_le(fmt, 16);
    fmt.resize(fmt_size, 0);
    append_chunk(out, "fmt ", fmt);

    std::vector<uint8_t> data(data_size, 0);
    append_chunk(out, "data", data);
    set_u32_le(out, 4, static_cast<uint32_t>(out.size() - 8));
    return out;
}

void save_binary_file(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    assert(f.good());
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(f.good());
}

bool can_read_wav_bytes(const std::vector<uint8_t>& bytes) {
    const auto path = write_temp_path("_parser.wav");
    save_binary_file(path, bytes);
    WavData wav;
    const bool ok = load_wav_file(path, wav);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return ok;
}

std::vector<uint8_t> make_lac_with_single_mono_sample(int32_t sample, uint8_t bit_depth = 16) {
    BitWriter bw;
    FrameHeader hdr;
    hdr.version = 2;
    hdr.channels = 1;
    hdr.stereo_mode = 0;
    hdr.bit_depth = bit_depth;
    hdr.write(bw);
    bw.write_bits(1u, 32); // block count
    bw.write_bits(1u, 32); // block size
    bw.write_bits(0u, 8); // fixed predictor
    bw.write_bits(0u, 8); // fixed order 0
    bw.write_bits(0u, 8); // residual control
    bw.write_bits(0u, 2); // Rice mode
    bw.write_bits(15u, 5);
    Rice::encode(bw, sample, 15u);
    bw.flush_to_byte();
    return bw.get_buffer();
}

std::vector<uint8_t> make_lac_header_only_with_block_count(uint32_t block_count) {
    BitWriter bw;
    FrameHeader hdr;
    hdr.version = 2;
    hdr.channels = 1;
    hdr.stereo_mode = 0;
    hdr.write(bw);
    bw.write_bits(block_count, 32);
    bw.flush_to_byte();
    return bw.get_buffer();
}

std::vector<uint8_t> make_lac_with_short_non_final_block() {
    BitWriter bw;
    FrameHeader hdr;
    hdr.version = 2;
    hdr.channels = 1;
    hdr.stereo_mode = 0;
    hdr.write(bw);
    bw.write_bits(2u, 32);
    bw.write_bits(Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE - 1u, 32);
    bw.write_bits(1u, 32);
    bw.flush_to_byte();
    return bw.get_buffer();
}

std::vector<uint8_t> make_lac_over_allocation_limit() {
    constexpr uint32_t blocks = 16385;
    BitWriter bw;
    FrameHeader hdr;
    hdr.version = 2;
    hdr.channels = 1;
    hdr.stereo_mode = 0;
    hdr.write(bw);
    bw.write_bits(blocks, 32);
    for (uint32_t i = 0; i < blocks; ++i) {
        bw.write_bits(Block::MAX_BLOCK_SIZE, 32);
    }
    bw.flush_to_byte();
    return bw.get_buffer();
}

std::vector<uint8_t> make_v3_mono_stream(size_t frames) {
    std::vector<int32_t> pcm(frames, 0);
    for (size_t i = 0; i < frames; ++i) {
        pcm[i] = static_cast<int32_t>((i % 257u) - 128);
    }
    LAC::Encoder encoder(12, 0, 44100, 16, false, false, false);
    encoder.set_thread_count(2);
    return encoder.encode(pcm, {});
}

} // namespace


void run_decoder_error_tests() {
    LAC::Decoder decoder;
    std::vector<int32_t> out_left, out_right;
    FrameHeader hdr;

    auto expect_throw = [&](const char* label, const uint8_t* data, size_t size) {
        out_left = {1};
        out_right = {1};
        bool threw = false;
        try {
            decoder.decode(data, size, out_left, out_right, &hdr);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "[decoder-error-test] expected throw: " << label << "\n";
            assert(false);
        }
        assert(out_left.empty());
        assert(out_right.empty());
    };

    std::vector<uint8_t> empty;
    expect_throw("empty-data", empty.data(), empty.size());

    std::vector<uint8_t> invalid_header(10, 0);
    expect_throw("invalid-header", invalid_header.data(), invalid_header.size());

    BitWriter bw;
    FrameHeader valid_header;
    valid_header.write(bw);
    bw.flush_to_byte();
    std::vector<uint8_t> header_only = bw.get_buffer();
    expect_throw("header-only", header_only.data(), header_only.size());

    std::vector<uint8_t> reserved_header = make_lac_with_single_mono_sample(0);
    reserved_header[9] = 1;
    expect_throw("reserved-frame-header", reserved_header.data(), reserved_header.size());

    std::vector<uint8_t> trailing_payload = make_lac_with_single_mono_sample(0);
    trailing_payload.push_back(0);
    expect_throw("trailing-payload", trailing_payload.data(), trailing_payload.size());

    std::vector<uint8_t> truncated_table = make_lac_header_only_with_block_count(2);
    expect_throw("truncated-block-size-table", truncated_table.data(), truncated_table.size());

    std::vector<uint8_t> excessive_block_count =
        make_lac_header_only_with_block_count(1'048'577u);
    expect_throw("excessive-block-count", excessive_block_count.data(), excessive_block_count.size());

    std::vector<uint8_t> short_non_final_block = make_lac_with_short_non_final_block();
    expect_throw("short-non-final-block", short_non_final_block.data(), short_non_final_block.size());

    std::vector<uint8_t> oversized_allocation = make_lac_over_allocation_limit();
    expect_throw("decoded-allocation-limit", oversized_allocation.data(), oversized_allocation.size());

    std::vector<uint8_t> out_of_range = make_lac_with_single_mono_sample(32768);
    expect_throw("decoded-sample-out-of-range", out_of_range.data(), out_of_range.size());

    LAC::Encoder stereo_encoder(12, 2, 44100, 16, false, false, false);
    std::vector<uint8_t> stereo_stream = stereo_encoder.encode({0}, {0});
    stereo_stream[v3_payload_offset(stereo_stream)] = 2;
    expect_throw("invalid-per-block-stereo-flag", stereo_stream.data(), stereo_stream.size());

    std::vector<uint8_t> unknown_version = make_v3_mono_stream(4096);
    unknown_version[2] = 4;
    expect_throw("unknown-frame-version", unknown_version.data(), unknown_version.size());

    std::vector<uint8_t> truncated_v3_table = make_v3_mono_stream(4096);
    truncated_v3_table.resize(14u + 7u);
    expect_throw("truncated-v3-table", truncated_v3_table.data(), truncated_v3_table.size());

    std::vector<uint8_t> zero_compressed_size = make_v3_mono_stream(4096);
    set_u32_be(zero_compressed_size, 18, 0);
    expect_throw("zero-compressed-block-size", zero_compressed_size.data(), zero_compressed_size.size());

    std::vector<uint8_t> mismatched_compressed_sum = make_v3_mono_stream(4096);
    const uint32_t original_size = get_u32_be(mismatched_compressed_sum, 18);
    set_u32_be(mismatched_compressed_sum, 18, original_size + 1u);
    expect_throw("compressed-size-sum-mismatch", mismatched_compressed_sum.data(), mismatched_compressed_sum.size());

    std::vector<uint8_t> crossed_block_boundary = make_v3_mono_stream(32768);
    assert(get_u32_be(crossed_block_boundary, 10) >= 2u);
    const uint32_t first_payload_size = get_u32_be(crossed_block_boundary, 18);
    const uint32_t second_payload_size = get_u32_be(crossed_block_boundary, 26);
    assert(first_payload_size > 1u);
    set_u32_be(crossed_block_boundary, 18, first_payload_size - 1u);
    set_u32_be(crossed_block_boundary, 26, second_payload_size + 1u);
    expect_throw("crossed-v3-block-boundary", crossed_block_boundary.data(), crossed_block_boundary.size());

    std::vector<uint8_t> extra_byte_in_block = make_v3_mono_stream(32768);
    const size_t payload_offset = v3_payload_offset(extra_byte_in_block);
    const uint32_t first_span = get_u32_be(extra_byte_in_block, 18);
    extra_byte_in_block.insert(extra_byte_in_block.begin() + payload_offset + first_span, 0u);
    set_u32_be(extra_byte_in_block, 18, first_span + 1u);
    expect_throw("extra-byte-inside-v3-block", extra_byte_in_block.data(), extra_byte_in_block.size());

    std::cout << "decoder error tests ok\n";
}

void run_decoder_thread_tests() {
    constexpr size_t frames = 65536;
    std::vector<int32_t> input(frames, 0);
    for (size_t i = 0; i < frames; ++i) {
        input[i] = static_cast<int32_t>((i % 2000u) - 1000);
    }

    LAC::Encoder encoder(12, 0, 44100, 16, false, false, false);
    encoder.set_thread_count(2);
    const std::vector<uint8_t> v3_stream = encoder.encode(input, {});
    assert(v3_stream[2] == 3u);
    assert(get_u32_be(v3_stream, 10) > 1u);

    LAC::ThreadCollector v3_collector;
    LAC::Decoder v3_decoder(&v3_collector);
    v3_decoder.set_thread_count(2);
    std::vector<int32_t> out_left;
    std::vector<int32_t> out_right;
    FrameHeader hdr;
    v3_decoder.decode(v3_stream.data(), v3_stream.size(), out_left, out_right, &hdr);
    assert(out_left == input);
    assert(out_right.empty());
    assert(hdr.version == 3u);
    const size_t v3_threads = v3_collector.snapshot().size();
    assert(v3_threads >= 1u);
    assert(v3_threads <= 2u);

    const std::vector<uint8_t> v2_stream = make_lac_with_single_mono_sample(123);
    LAC::ThreadCollector v2_collector;
    LAC::Decoder v2_decoder(&v2_collector);
    v2_decoder.set_thread_count(2);
    v2_decoder.decode(v2_stream.data(), v2_stream.size(), out_left, out_right, &hdr);
    assert(out_left == std::vector<int32_t>{123});
    assert(out_right.empty());
    assert(hdr.version == 2u);
    assert(v2_collector.snapshot().size() == 1u);
    std::cout << "decoder thread tests ok\n";
}

void run_block_planner_tests() {
    auto assert_sizes = [](size_t frames, const std::vector<uint32_t>& expected) {
        LAC::Encoder encoder(12, 0, 44100, 16, false, false, false);
        const std::vector<uint8_t> stream = encoder.encode(std::vector<int32_t>(frames, 0), {});
        assert(v3_block_sizes(stream) == expected);
    };

    assert_sizes(16383, {16383});
    assert_sizes(16384, {16384});
    assert_sizes(16385, {16384, 1});
    assert_sizes(65536, {16384, 16384, 16384, 16384});

    std::vector<int32_t> quarters(Block::MAX_BLOCK_SIZE, 0);
    for (size_t i = Block::MAX_BLOCK_SIZE / 4u; i < Block::MAX_BLOCK_SIZE / 2u; ++i) {
        quarters[i] = 1000;
    }
    for (size_t i = Block::MAX_BLOCK_SIZE / 2u; i < 3u * Block::MAX_BLOCK_SIZE / 4u; ++i) {
        quarters[i] = -1000;
    }
    LAC::Encoder encoder(12, 0, 44100, 16, false, false, false);
    const std::vector<uint8_t> quarters_stream = encoder.encode(quarters, {});
    assert(v3_block_sizes(quarters_stream) == std::vector<uint32_t>{Block::MAX_BLOCK_SIZE});
    Block::Encoder block_encoder(12);
    block_encoder.set_zero_run_enabled(true);
    block_encoder.set_partitioning_enabled(true);
    const std::vector<uint8_t> quarters_block = block_encoder.encode(quarters);
    assert(quarters_stream.size() == 22u + quarters_block.size());

    std::vector<int32_t> silence_then_noise(Block::MAX_BLOCK_SIZE);
    uint32_t state = 29u * 8191u;
    for (size_t i = 0; i < silence_then_noise.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        const int32_t sample = static_cast<int32_t>(state >> 16) - 32768;
        silence_then_noise[i] = i < silence_then_noise.size() / 2u ? 0 : sample;
    }
    const std::vector<uint8_t> silence_then_noise_stream = encoder.encode(silence_then_noise, {});
    assert(v3_block_sizes(silence_then_noise_stream) == std::vector<uint32_t>{Block::MAX_BLOCK_SIZE});
    const std::vector<uint8_t> silence_then_noise_block = block_encoder.encode(silence_then_noise);
    assert(silence_then_noise_stream.size() == 22u + silence_then_noise_block.size());
    std::cout << "block planner tests ok\n";
}

void run_stereo_planner_tests() {
    constexpr size_t block = Block::MAX_BLOCK_SIZE;
    std::vector<int32_t> left(block * 2u);
    std::vector<int32_t> right(block * 2u);
    for (size_t i = 0; i < left.size(); ++i) {
        left[i] = static_cast<int32_t>((i % 2001u) - 1000);
        right[i] = (i < block) ? left[i] : 0;
    }

    LAC::Encoder encoder(12, 2, 44100, 16, false, false, false);
    encoder.set_thread_count(2);
    const std::vector<uint8_t> mixed_stream = encoder.encode(left, right);
    const std::vector<uint8_t> mixed_flags = v3_stereo_flags(mixed_stream);
    assert(std::find(mixed_flags.begin(), mixed_flags.end(), 0u) != mixed_flags.end());
    assert(std::find(mixed_flags.begin(), mixed_flags.end(), 1u) != mixed_flags.end());

    std::vector<int32_t> identical(block, 0);
    for (size_t i = 0; i < identical.size(); ++i) {
        identical[i] = static_cast<int32_t>((i % 2001u) - 1000);
    }
    const std::vector<uint8_t> ms_stream = encoder.encode(identical, identical);
    const std::vector<uint8_t> ms_flags = v3_stereo_flags(ms_stream);
    assert(!ms_flags.empty());
    assert(std::all_of(ms_flags.begin(), ms_flags.end(), [](uint8_t flag) { return flag == 1u; }));

    const std::vector<uint8_t> lr_stream = encoder.encode(identical, std::vector<int32_t>(block, 0));
    const std::vector<uint8_t> lr_flags = v3_stereo_flags(lr_stream);
    assert(!lr_flags.empty());
    assert(std::all_of(lr_flags.begin(), lr_flags.end(), [](uint8_t flag) { return flag == 0u; }));

    std::vector<int32_t> anticorrelated(4096);
    std::vector<int32_t> inverse(4096);
    uint32_t state = 1;
    for (size_t i = 0; i < anticorrelated.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        anticorrelated[i] = static_cast<int32_t>(state >> 18) - 8192;
        inverse[i] = -anticorrelated[i];
    }
    anticorrelated[0] = -32768;
    inverse[0] = 32767;

    LAC::Encoder auto_encoder(12, 2, 44100, 16, false, false, false);
    LAC::Encoder forced_lr(12, 0, 44100, 16, false, false, false);
    LAC::Encoder forced_ms(12, 1, 44100, 16, false, false, false);
    auto_encoder.set_thread_count(1);
    forced_lr.set_thread_count(1);
    forced_ms.set_thread_count(1);
    const std::vector<uint8_t> auto_stream = auto_encoder.encode(anticorrelated, inverse);
    const std::vector<uint8_t> forced_lr_stream = forced_lr.encode(anticorrelated, inverse);
    const std::vector<uint8_t> forced_ms_stream = forced_ms.encode(anticorrelated, inverse);
    const std::vector<uint8_t> auto_flags = v3_stereo_flags(auto_stream);
    assert(std::all_of(auto_flags.begin(), auto_flags.end(), [](uint8_t flag) { return flag == 1u; }));
    assert(auto_stream.size() == forced_ms_stream.size() + auto_flags.size());
    assert(auto_stream.size() < forced_lr_stream.size());

    std::vector<int32_t> random_left(4096);
    std::vector<int32_t> random_right(4096);
    state = 49917u;
    for (size_t i = 0; i < random_left.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        random_left[i] = static_cast<int32_t>(state >> 16) - 32768;
        state = state * 1664525u + 1013904223u;
        random_right[i] = static_cast<int32_t>(state >> 16) - 32768;
    }
    const std::vector<uint8_t> auto_random_stream = auto_encoder.encode(random_left, random_right);
    const std::vector<uint8_t> lr_random_stream = forced_lr.encode(random_left, random_right);
    const std::vector<uint8_t> ms_random_stream = forced_ms.encode(random_left, random_right);
    assert_per_block_stereo_payload_consistency(auto_random_stream, lr_random_stream, ms_random_stream);

    std::vector<int32_t> alternating_left(4095);
    std::vector<int32_t> alternating_right(4095);
    state = 758392u;
    for (size_t i = 0; i < alternating_left.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        state = state * 1664525u + 1013904223u;
        const int32_t noise = static_cast<int32_t>(state >> 16) - 32768;
        alternating_left[i] = (i & 1u) ? static_cast<int32_t>(i) : -static_cast<int32_t>(i);
        alternating_right[i] = alternating_left[i] + noise % 257;
    }
    const std::vector<uint8_t> auto_alternating_stream = auto_encoder.encode(alternating_left, alternating_right);
    const std::vector<uint8_t> lr_alternating_stream = forced_lr.encode(alternating_left, alternating_right);
    const std::vector<uint8_t> ms_alternating_stream = forced_ms.encode(alternating_left, alternating_right);
    const std::vector<uint8_t> auto_alternating_flags = v3_stereo_flags(auto_alternating_stream);
    assert(std::all_of(auto_alternating_flags.begin(), auto_alternating_flags.end(), [](uint8_t flag) { return flag == 0u; }));
    assert(auto_alternating_stream.size() == lr_alternating_stream.size() + auto_alternating_flags.size());
    assert(auto_alternating_stream.size() < ms_alternating_stream.size());

    std::vector<int32_t> walk_left(4095);
    std::vector<int32_t> walk_right(4095);
    state = 13210319u;
    int32_t left_walk = 0;
    int32_t right_walk = 0;
    for (size_t i = 0; i < walk_left.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        const int32_t x = static_cast<int32_t>(state >> 16) - 32768;
        state = state * 1664525u + 1013904223u;
        const int32_t y = static_cast<int32_t>(state >> 16) - 32768;
        left_walk = std::clamp<int32_t>(left_walk + x % 65, -32768, 32767);
        right_walk = std::clamp<int32_t>(right_walk + y % 65, -32768, 32767);
        walk_left[i] = left_walk;
        walk_right[i] = right_walk;
    }
    const std::vector<uint8_t> auto_walk_stream = auto_encoder.encode(walk_left, walk_right);
    const std::vector<uint8_t> lr_walk_stream = forced_lr.encode(walk_left, walk_right);
    const std::vector<uint8_t> ms_walk_stream = forced_ms.encode(walk_left, walk_right);
    assert_per_block_stereo_payload_consistency(auto_walk_stream, lr_walk_stream, ms_walk_stream);

    std::vector<int32_t> long_walk_left(Block::MAX_BLOCK_SIZE);
    std::vector<int32_t> long_walk_right(Block::MAX_BLOCK_SIZE);
    state = 13210319u;
    left_walk = 0;
    right_walk = 0;
    for (size_t i = 0; i < long_walk_left.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        const int32_t x = static_cast<int32_t>(state >> 16) - 32768;
        state = state * 1664525u + 1013904223u;
        const int32_t y = static_cast<int32_t>(state >> 16) - 32768;
        left_walk = std::clamp<int32_t>(left_walk + x % 65, -32768, 32767);
        right_walk = std::clamp<int32_t>(right_walk + y % 65, -32768, 32767);
        long_walk_left[i] = left_walk;
        long_walk_right[i] = right_walk;
    }
    const std::vector<uint8_t> auto_long_walk_stream = auto_encoder.encode(long_walk_left, long_walk_right);
    const std::vector<uint8_t> lr_long_walk_stream = forced_lr.encode(long_walk_left, long_walk_right);
    const std::vector<uint8_t> ms_long_walk_stream = forced_ms.encode(long_walk_left, long_walk_right);
    assert_per_block_stereo_payload_consistency(auto_long_walk_stream, lr_long_walk_stream, ms_long_walk_stream);

    std::vector<int32_t> noise_left(Block::MAX_BLOCK_SIZE);
    std::vector<int32_t> noise_right(Block::MAX_BLOCK_SIZE);
    state = 1u;
    for (size_t i = 0; i < noise_left.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        noise_left[i] = static_cast<int32_t>(state >> 16) - 32768;
        state = state * 1664525u + 1013904223u;
        noise_right[i] = static_cast<int32_t>(state >> 16) - 32768;
    }
    const std::vector<uint8_t> auto_noise_stream = auto_encoder.encode(noise_left, noise_right);
    const std::vector<uint8_t> lr_noise_stream = forced_lr.encode(noise_left, noise_right);
    const std::vector<uint8_t> ms_noise_stream = forced_ms.encode(noise_left, noise_right);
    const std::vector<uint8_t> auto_noise_flags = v3_stereo_flags(auto_noise_stream);
    assert(auto_noise_flags.size() == 1u);
    const std::vector<uint8_t>& selected_noise_stream = auto_noise_flags[0] == 1u ? ms_noise_stream : lr_noise_stream;
    assert(auto_noise_stream.size() == selected_noise_stream.size() + auto_noise_flags.size());
    std::cout << "stereo planner tests ok\n";
}

void run_wav_validation_tests() {
    assert(can_read_wav_bytes(make_pcm16_mono_wav(true)));

    std::vector<uint8_t> short_fmt = make_pcm16_mono_wav(false, 14);
    assert(!can_read_wav_bytes(short_fmt));

    std::vector<uint8_t> extended_fmt = make_pcm16_mono_wav(false, 18);
    assert(!can_read_wav_bytes(extended_fmt));

    std::vector<uint8_t> bad_riff_size = make_pcm16_mono_wav();
    set_u32_le(bad_riff_size, 4, static_cast<uint32_t>(bad_riff_size.size() - 9));
    assert(!can_read_wav_bytes(bad_riff_size));

    std::vector<uint8_t> truncated_data = make_pcm16_mono_wav();
    truncated_data.pop_back();
    set_u32_le(truncated_data, 4, static_cast<uint32_t>(truncated_data.size() - 8));
    assert(!can_read_wav_bytes(truncated_data));

    std::vector<uint8_t> misaligned_data = make_pcm16_mono_wav(false, 16, 1);
    assert(!can_read_wav_bytes(misaligned_data));

    std::vector<uint8_t> bad_byte_rate = make_pcm16_mono_wav();
    set_u32_le(bad_byte_rate, 28, 1);
    assert(!can_read_wav_bytes(bad_byte_rate));

    std::vector<uint8_t> bad_audio_format = make_pcm16_mono_wav();
    set_u16_le(bad_audio_format, 20, 3);
    assert(!can_read_wav_bytes(bad_audio_format));

    std::vector<uint8_t> bad_channels = make_pcm16_mono_wav();
    set_u16_le(bad_channels, 22, 3);
    assert(!can_read_wav_bytes(bad_channels));

    std::vector<uint8_t> bad_sample_rate = make_pcm16_mono_wav();
    set_u32_le(bad_sample_rate, 24, 12345);
    assert(!can_read_wav_bytes(bad_sample_rate));

    std::vector<uint8_t> bad_block_align = make_pcm16_mono_wav();
    set_u16_le(bad_block_align, 32, 1);
    assert(!can_read_wav_bytes(bad_block_align));

    std::vector<uint8_t> bad_bit_depth = make_pcm16_mono_wav();
    set_u16_le(bad_bit_depth, 34, 8);
    assert(!can_read_wav_bytes(bad_bit_depth));

    assert(!can_read_wav_bytes(make_pcm16_mono_wav(false, 16, 0)));

    std::vector<uint8_t> duplicate_fmt = make_pcm16_mono_wav();
    append_chunk(duplicate_fmt, "fmt ", std::vector<uint8_t>(16, 0));
    set_u32_le(duplicate_fmt, 4, static_cast<uint32_t>(duplicate_fmt.size() - 8));
    assert(!can_read_wav_bytes(duplicate_fmt));

    std::vector<uint8_t> duplicate_data = make_pcm16_mono_wav();
    append_chunk(duplicate_data, "data", {0, 0});
    set_u32_le(duplicate_data, 4, static_cast<uint32_t>(duplicate_data.size() - 8));
    assert(!can_read_wav_bytes(duplicate_data));

    const auto output_path = write_temp_path("_writer.wav");
    assert(!write_wav(output_path.string(), {}, {}, 1, 44100, 16));
    assert(!write_wav(output_path.string(), {32768}, {}, 1, 44100, 16));
    assert(!write_wav(output_path.string(), {0}, {0}, 1, 44100, 16));
    assert(write_wav(output_path.string(), {1}, {}, 1, 44100, 24));
    const auto odd_pcm24 = load_binary_file(output_path);
    assert(odd_pcm24.size() == 48u);
    assert(get_u32_le(odd_pcm24, 4) == 40u);
    assert(get_u32_le(odd_pcm24, 40) == 3u);
    assert(odd_pcm24.back() == 0u);
    WavData odd_pcm24_restored;
    assert(load_wav_file(output_path, odd_pcm24_restored));
    assert(odd_pcm24_restored.left == std::vector<int32_t>{1});
    std::error_code ec;
    std::filesystem::remove(output_path, ec);

    std::cout << "wav validation tests ok\n";
}

void run_encoder_validation_tests() {
    const std::vector<int32_t> mono(16, 0);
    const std::vector<int32_t> right(16, 0);
    const std::vector<int32_t> short_right(8, 0);

    auto expect_invalid = [](const char* label, LAC::Encoder& encoder,
                             const std::vector<int32_t>& left,
                             const std::vector<int32_t>& right_channel) {
        bool threw = false;
        try {
            (void)encoder.encode(left, right_channel);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "[encoder-validation-miss] " << label << "\n";
        }
        assert(threw);
    };

    LAC::Encoder empty_input(12, 0, 44100, 16, false, false, false);
    expect_invalid("empty input", empty_input, {}, {});

    LAC::Encoder mismatched_channels(12, 0, 44100, 16, false, false, false);
    expect_invalid("mismatched channels", mismatched_channels, mono, short_right);

    LAC::Encoder bad_sample_rate(12, 0, 12345, 16, false, false, false);
    expect_invalid("bad sample rate", bad_sample_rate, mono, {});

    LAC::Encoder bad_bit_depth(12, 0, 44100, 20, false, false, false);
    expect_invalid("bad bit depth", bad_bit_depth, mono, {});

    LAC::Encoder bad_stereo_mode(12, 3, 44100, 16, false, false, false);
    expect_invalid("bad stereo mode", bad_stereo_mode, mono, right);

    LAC::Encoder out_of_range_16(12, 0, 44100, 16, false, false, false);
    expect_invalid("out-of-range 16-bit sample", out_of_range_16, {32768}, {});

    LAC::Encoder out_of_range_24(12, 0, 44100, 24, false, false, false);
    expect_invalid("out-of-range 24-bit sample", out_of_range_24, {0x800000}, {});

    std::cout << "encoder validation tests ok\n";
}

void run_thread_limit_tests() {
    constexpr size_t frames = 65536;
    std::vector<int32_t> left(frames);
    std::vector<int32_t> right(frames);
    for (size_t i = 0; i < frames; ++i) {
        left[i] = static_cast<int32_t>((i % 2000) - 1000);
        right[i] = static_cast<int32_t>(1000 - (i % 2000));
    }

    LAC::Encoder encoder(12, 0, 44100, 16, false, false, false);
    encoder.set_thread_count(2);
    LAC::ThreadCollector collector;
    const std::vector<uint8_t> bitstream = encoder.encode(left, right, &collector);
    assert(!bitstream.empty());

    const auto threads = collector.snapshot();
    assert(!threads.empty());
    assert(threads.size() <= 2);
    std::cout << "thread limit tests ok\n";
}

void run_e2e_tests() {
    const char* files[] = {
        "16.44100.wav",
        "24.44100.wav",
        "24.48000.wav",
        "24.96000.wav",
        "24.192000.wav"
    };
    bool ran_asset_case = false;
    for (const char* f : files) {
        if (!std::filesystem::exists(assets_dir() / f)) {
            std::cout << "[e2e] skip missing asset " << f << "\n";
            continue;
        }
        run_case(f);
        ran_asset_case = true;
    }
    if (!ran_asset_case) {
        std::cout << "[e2e] no asset fixtures found; running generated fixtures\n";
    }
    run_generated_cases();
    std::cout << "e2e wav->lac->wav tests ok\n";
}
