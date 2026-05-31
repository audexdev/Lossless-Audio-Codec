#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"
#include "io/wav_io.hpp"
#include "codec/frame/frame_header.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

void run_case_path(const std::filesystem::path& src_path, const std::string& label) {
    std::cout << "[e2e] start " << label << "\n";
    WavData orig;
    bool ok = load_wav_file(src_path, orig);
    assert(ok && "Failed to load source WAV");

    uint8_t stereo_mode = (orig.channels == 2) ? 2 : 0;
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
    for (size_t i = 0; i < frames; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(sample_rate);
        left[i] = static_cast<int32_t>(std::sin(2.0 * 3.14159265358979323846 * 440.0 * t) * amplitude);
        if (channels == 2) {
            right[i] = static_cast<int32_t>(std::sin(2.0 * 3.14159265358979323846 * 661.0 * t) * (amplitude / 2));
        }
    }

    const auto wav_path = write_temp_path("_" + label + ".wav");
    const bool ok = write_wav(wav_path.string(), left, right, channels, sample_rate, bit_depth);
    assert(ok && "Failed to write generated WAV fixture");
    return wav_path;
}

void run_generated_cases() {
    const auto mono_path = write_generated_wav("generated_mono_16_44100", 1, 44100, 16, 4096);
    const auto stereo_path = write_generated_wav("generated_stereo_24_48000", 2, 48000, 24, 4096);

    run_case_path(mono_path, "generated_mono_16_44100.wav");
    run_case_path(stereo_path, "generated_stereo_24_48000.wav");

    std::error_code ec;
    std::filesystem::remove(mono_path, ec);
    std::filesystem::remove(stereo_path, ec);
}

} // namespace


void run_decoder_error_tests() {
    LAC::Decoder decoder;
    std::vector<int32_t> out_left, out_right;
    FrameHeader hdr;

    auto expect_throw = [&](const char* label, const uint8_t* data, size_t size) {
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

    std::cout << "decoder error tests ok\n";
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
