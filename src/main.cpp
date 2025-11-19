#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
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
    std::cerr << "  lac_cli encode input.wav output.lac [--stereo-mode=lr|ms]\n";
    std::cerr << "  lac_cli decode input.lac output.wav\n";
    std::cerr << "  lac_cli selftest\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "encode") {
        if (argc < 4 || argc > 5) {
            usage();
            return 1;
        }
        std::string in_path = argv[2];
        std::string out_path = argv[3];
        uint8_t stereo_mode = 0;
        if (argc == 5) {
            std::string flag = argv[4];
            if (flag == "--stereo-mode=lr") stereo_mode = 0;
            else if (flag == "--stereo-mode=ms") stereo_mode = 1;
            else {
                usage();
                return 1;
            }
        }
        std::vector<int32_t> left, right;
        uint16_t channels = 0;
        uint32_t sample_rate = 0;
        if (!read_wav(in_path, left, right, channels, sample_rate)) {
            std::cerr << "Failed to read WAV: " << in_path << "\n";
            return 1;
        }
        LAC::Encoder encoder(1024, 4, stereo_mode);
        std::vector<uint8_t> bitstream = encoder.encode(left, right);
        if (!save_file(out_path, bitstream)) {
            std::cerr << "Failed to write LAC file: " << out_path << "\n";
            return 1;
        }
        std::cout << "Encoded " << in_path << " -> " << out_path << " (" << bitstream.size() << " bytes)\n";
        return 0;
    }

    if (mode == "decode") {
        if (argc != 4) {
            usage();
            return 1;
        }
        std::string in_path = argv[2];
        std::string out_path = argv[3];
        std::vector<uint8_t> bitstream;
        if (!load_file(in_path, bitstream)) {
            std::cerr << "Failed to read LAC file: " << in_path << "\n";
            return 1;
        }
        LAC::Decoder decoder;
        std::vector<int32_t> left, right;
        decoder.decode(bitstream.data(), bitstream.size(), left, right);
        if (left.empty()) {
            std::cerr << "Decode failed or produced no samples\n";
            return 1;
        }
        uint16_t channels = right.empty() ? 1 : 2;
        if (!write_wav(out_path, left, right, channels, 44100)) {
            std::cerr << "Failed to write WAV: " << out_path << "\n";
            return 1;
        }
        std::cout << "Decoded " << in_path << " -> " << out_path << " (" << left.size() << " samples per channel)\n";
        return 0;
    }

    if (mode == "selftest") {
        const uint32_t sample_rate = 44100;
        const size_t total_samples = 44100;
        std::vector<int32_t> left(total_samples);
        std::vector<int32_t> right(total_samples);
        const double pi = 3.14159265358979323846;
        for (size_t i = 0; i < total_samples; ++i) {
            double t = static_cast<double>(i) / sample_rate;
            left[i] = static_cast<int32_t>(std::sin(2.0 * pi * 440.0 * t) * 20000);
            right[i] = static_cast<int32_t>(std::sin(2.0 * pi * 441.0 * t) * 19000);
        }

        LAC::Encoder enc_lr(1024, 4, 0);
        std::vector<uint8_t> bs_lr = enc_lr.encode(left, right);

        LAC::Encoder enc_ms(1024, 4, 1);
        std::vector<uint8_t> bs_ms = enc_ms.encode(left, right);

        LAC::Decoder dec;
        std::vector<int32_t> dlr_l, dlr_r;
        auto t0 = std::chrono::high_resolution_clock::now();
        dec.decode(bs_lr.data(), bs_lr.size(), dlr_l, dlr_r);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> dms_l, dms_r;
        auto t2 = std::chrono::high_resolution_clock::now();
        dec.decode(bs_ms.data(), bs_ms.size(), dms_l, dms_r);
        auto t3 = std::chrono::high_resolution_clock::now();

        if (dlr_l != left || dlr_r != right) {
            std::cerr << "LR roundtrip mismatch\n";
            return 1;
        }
        if (dms_l != left || dms_r != right) {
            std::cerr << "MS roundtrip mismatch\n";
            return 1;
        }
        bool smaller = bs_ms.size() < bs_lr.size();
        auto lr_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto ms_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        std::cout << "Selftest OK. LR size=" << bs_lr.size() << " bytes, MS size=" << bs_ms.size() << " bytes"
                  << " (MS " << (smaller ? "smaller" : "not smaller") << "), decode us: LR=" << lr_us
                  << " MS=" << ms_us << "\n";
        return 0;
    }

    usage();
    return 1;
}
