#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"
#include "io/wav_io.hpp"
#include "codec/frame/frame_header.hpp"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
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

void run_case(const std::string& filename) {
    const auto src_path = assets_dir() / filename;
    std::cout << "[e2e] start " << filename << "\n";
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
    }

    LAC::Decoder decoder;
    std::vector<int32_t> dec_left, dec_right;
    FrameHeader hdr;
    ok = decoder.decode(bitstream.data(), bitstream.size(), dec_left, dec_right, &hdr);
    assert(ok && "Decode failed");

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
        std::cerr << "[e2e-mismatch] file=" << filename
                  << " left_mis=" << left_cmp.mismatches << " left_maxdiff=" << left_cmp.max_abs_diff
                  << " right_mis=" << right_cmp.mismatches << " right_maxdiff=" << right_cmp.max_abs_diff
                  << "\n";
        assert(false && "PCM mismatch");
    }

    std::error_code ec;
    std::filesystem::remove(lac_path, ec);
    std::filesystem::remove(out_wav_path, ec);
    std::cout << "[e2e] ok " << filename << "\n";
}

} // namespace

void run_e2e_tests() {
    const char* files[] = {
        "16.44100.wav",
        "24.44100.wav",
        "24.48000.wav",
        "24.96000.wav",
        "24.192000.wav"
    };
    for (const char* f : files) {
        run_case(f);
    }
    std::cout << "e2e wav->lac->wav tests ok\n";
}
