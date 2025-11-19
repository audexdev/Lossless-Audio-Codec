#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "io/wav_io.hpp"
#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"

static bool load_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));
    if (size > 0) {
        f.read(reinterpret_cast<char*>(data.data()), size);
        if (f.gcount() != size) return false;
    }
    return true;
}

static bool save_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return f.good();
}

static void usage() {
    std::cerr << "Usage:\n";
    std::cerr << "  lac_cli encode input.wav output.lac [--stereo-mode=lr|ms] [--debug-threads]\n";
    std::cerr << "  lac_cli decode input.lac output.wav [--debug-threads]\n";
    std::cerr << "  lac_cli selftest\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "encode") {
        if (argc < 4) {
            usage();
            return 1;
        }
        std::string in_path = argv[2];
        std::string out_path = argv[3];
        uint8_t stereo_mode = 0;
        bool debug_threads = false;
        for (int i = 4; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--debug-threads") {
                debug_threads = true;
            } else if (flag == "--stereo-mode=lr") {
                stereo_mode = 0;
            } else if (flag == "--stereo-mode=ms") {
                stereo_mode = 1;
            } else {
                usage();
                return 1;
            }
        }
        std::vector<int32_t> left, right;
        uint16_t channels = 0;
        uint32_t sample_rate = 0;
        uint8_t bit_depth = 0;
        if (!read_wav(in_path, left, right, channels, sample_rate, bit_depth)) {
            std::cerr << "Failed to read WAV: " << in_path << "\n";
            return 1;
        }
        LAC::Encoder encoder(4, stereo_mode, sample_rate, bit_depth);
        LAC::ThreadCollector collector;
        LAC::ThreadCollector* collector_ptr = (debug_threads ? &collector : nullptr);
        std::vector<uint8_t> bitstream = encoder.encode(left, right, collector_ptr);
        if (!save_file(out_path, bitstream)) {
            std::cerr << "Failed to write LAC file: " << out_path << "\n";
            return 1;
        }
        std::cout << "Encoded " << in_path << " -> " << out_path << " (" << bitstream.size() << " bytes)\n";
        if (debug_threads) {
            auto threads = collector.snapshot();
            std::cout << "Thread usage: " << threads.size() << " threads\n";
            for (const auto& tid : threads) {
                std::cout << "  " << tid << "\n";
            }
            if (threads.size() <= 1) {
                std::cout << "WARNING: Multi-threading not active (single-threaded execution).\n";
            }
        }
        return 0;
    }

    if (mode == "decode") {
        if (argc < 4) {
            usage();
            return 1;
        }
        std::string in_path = argv[2];
        std::string out_path = argv[3];
        bool debug_threads = false;
        for (int i = 4; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--debug-threads") {
                debug_threads = true;
            } else {
                usage();
                return 1;
            }
        }
        std::vector<uint8_t> bitstream;
        if (!load_file(in_path, bitstream)) {
            std::cerr << "Failed to read LAC file: " << in_path << "\n";
            return 1;
        }
        LAC::ThreadCollector decoderCollector;
        LAC::ThreadCollector* decoderCollectorPtr = (debug_threads ? &decoderCollector : nullptr);
        LAC::Decoder decoder(decoderCollectorPtr);
        std::vector<int32_t> left, right;
        FrameHeader hdr;
        if (!decoder.decode(bitstream.data(), bitstream.size(), left, right, &hdr) || left.empty()) {
            std::cerr << "Decode failed or produced no samples\n";
            return 1;
        }
        if (!write_wav(out_path, left, right, hdr.channels, hdr.sample_rate, hdr.bit_depth)) {
            std::cerr << "Failed to write WAV: " << out_path << "\n";
            return 1;
        }
        std::cout << "Decoded " << in_path << " -> " << out_path << " (" << left.size() << " samples per channel)\n";
        if (debug_threads) {
            auto threads = decoderCollector.snapshot();
            std::cout << "Decoder thread usage: " << threads.size() << " threads\n";
            for (const auto& tid : threads) {
                std::cout << "  " << tid << "\n";
            }
            if (threads.size() <= 1) {
                std::cout << "WARNING: Decoder multi-threading may not be active.\n";
            }
        }
        return 0;
    }

    if (mode == "selftest") {
        const double pi = 3.14159265358979323846;
        constexpr int32_t pcm24_max = 0x7FFFFF;

        auto generate_signal = [&](uint32_t sample_rate, uint8_t bit_depth, size_t frames,
                                   std::vector<int32_t>& left, std::vector<int32_t>& right) {
            left.resize(frames);
            right.resize(frames);
            const int64_t amplitude = (bit_depth == 24)
                ? static_cast<int64_t>(pcm24_max) / 3
                : static_cast<int64_t>(30000);
            for (size_t i = 0; i < frames; ++i) {
                double t = static_cast<double>(i) / static_cast<double>(sample_rate);
                left[i] = static_cast<int32_t>(std::sin(2.0 * pi * 440.0 * t) * amplitude);
                right[i] = static_cast<int32_t>(std::sin(2.0 * pi * 443.0 * t) * (amplitude * 0.95));
            }
        };

        auto run_pair = [&](uint32_t sample_rate, uint8_t bit_depth) -> bool {
            const size_t frames = std::max<size_t>(sample_rate / 20, 2048);
            std::vector<int32_t> src_left;
            std::vector<int32_t> src_right;
            generate_signal(sample_rate, bit_depth, frames, src_left, src_right);

            LAC::Encoder enc_lr(4, 0, sample_rate, bit_depth);
            std::vector<uint8_t> bs_lr = enc_lr.encode(src_left, src_right);
            LAC::Decoder decoder;
            std::vector<int32_t> dec_lr_left, dec_lr_right;
            FrameHeader hdr_lr;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!decoder.decode(bs_lr.data(), bs_lr.size(), dec_lr_left, dec_lr_right, &hdr_lr)) {
                std::cerr << "LR decode failed for sr=" << sample_rate << " depth=" << int(bit_depth) << "\n";
                return false;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (dec_lr_left != src_left || dec_lr_right != src_right) {
                std::cerr << "LR roundtrip mismatch for sr=" << sample_rate << " depth=" << int(bit_depth) << "\n";
                return false;
            }
            if (hdr_lr.sample_rate != sample_rate || hdr_lr.bit_depth != bit_depth) {
                std::cerr << "LR header mismatch sr=" << hdr_lr.sample_rate << " depth=" << int(hdr_lr.bit_depth) << "\n";
                return false;
            }
            LAC::Encoder enc_ms(4, 1, sample_rate, bit_depth);
            std::vector<uint8_t> bs_ms = enc_ms.encode(src_left, src_right);
            std::vector<int32_t> dec_ms_left, dec_ms_right;
            FrameHeader hdr_ms;
            auto t2 = std::chrono::high_resolution_clock::now();
            if (!decoder.decode(bs_ms.data(), bs_ms.size(), dec_ms_left, dec_ms_right, &hdr_ms)) {
                std::cerr << "MS decode failed for sr=" << sample_rate << " depth=" << int(bit_depth) << "\n";
                return false;
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            if (dec_ms_left != src_left || dec_ms_right != src_right) {
                std::cerr << "MS roundtrip mismatch for sr=" << sample_rate << " depth=" << int(bit_depth) << "\n";
                return false;
            }
            if (hdr_ms.sample_rate != sample_rate || hdr_ms.bit_depth != bit_depth) {
                std::cerr << "MS header mismatch sr=" << hdr_ms.sample_rate << " depth=" << int(hdr_ms.bit_depth) << "\n";
                return false;
            }

            auto lr_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            auto ms_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            bool ms_smaller = bs_ms.size() < bs_lr.size();
            std::cout << "Selftest sr=" << sample_rate << "Hz depth=" << int(bit_depth)
                      << " LR=" << bs_lr.size() << " bytes (" << lr_us << "us decode)"
                      << " MS=" << bs_ms.size() << " bytes (" << ms_us << "us decode)"
                      << " -> MS is " << (ms_smaller ? "smaller" : "not smaller") << "\n";
            return true;
        };

        if (!run_pair(44100, 16)) return 1;
        if (!run_pair(48000, 24)) return 1;
        if (!run_pair(96000, 24)) return 1;
        if (!run_pair(192000, 24)) return 1;

        std::cout << "Selftest complete: adaptive block tests passed.\n";
        return 0;
    }

    usage();
    return 1;
}
