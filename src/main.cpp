#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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
    std::cerr << "  lac_cli encode input.wav output.lac\n";
    std::cerr << "  lac_cli decode input.lac output.wav\n";
}

int main(int argc, char** argv) {
    if (argc != 4) {
        usage();
        return 1;
    }

    std::string mode = argv[1];
    std::string in_path = argv[2];
    std::string out_path = argv[3];

    if (mode == "encode") {
        std::vector<int32_t> left, right;
        uint16_t channels = 0;
        uint32_t sample_rate = 0;
        if (!read_wav(in_path, left, right, channels, sample_rate)) {
            std::cerr << "Failed to read WAV: " << in_path << "\n";
            return 1;
        }
        LAC::Encoder encoder(1024, 4);
        std::vector<uint8_t> bitstream = encoder.encode(left, right);
        if (!save_file(out_path, bitstream)) {
            std::cerr << "Failed to write LAC file: " << out_path << "\n";
            return 1;
        }
        std::cout << "Encoded " << in_path << " -> " << out_path << " (" << bitstream.size() << " bytes)\n";
        return 0;
    }

    if (mode == "decode") {
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

    usage();
    return 1;
}
