#include <iostream>
#include <fstream>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif
#include "io/wav_io.hpp"
#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"

namespace {
constexpr uint64_t MAX_LAC_INPUT_BYTES = 1ULL << 30;
}

static bool load_file(const std::string& path, std::vector<uint8_t>& data) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamsize size = f.tellg();
  if (size < 0 || static_cast<uint64_t>(size) > MAX_LAC_INPUT_BYTES) return false;
  f.seekg(0, std::ios::beg);
  data.resize(static_cast<size_t>(size));
  if (size > 0) {
    f.read(reinterpret_cast<char*>(data.data()), size);
    if (f.gcount() != size) return false;
  }
  return true;
}

static bool save_file(const std::string& path, const std::vector<uint8_t>& data) {
  if (data.size() > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) return false;
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  f.close();
  return f.good();
}

static bool paths_refer_to_same_file(const std::string& input_path, const std::string& output_path) {
  std::error_code ec;
  if (std::filesystem::equivalent(input_path, output_path, ec)) {
    return true;
  }

  ec.clear();
  const auto normalized_input = std::filesystem::weakly_canonical(input_path, ec);
  if (ec) return false;
  const auto normalized_output = std::filesystem::weakly_canonical(output_path, ec);
  return !ec && normalized_input == normalized_output;
}

static std::string temporary_output_suffix() {
  static std::atomic<uint64_t> sequence{0};
  uint64_t token =
      static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  token ^= ++sequence;
  try {
    std::random_device random;
    token ^= (static_cast<uint64_t>(random()) << 32) ^ random();
  } catch (...) {
    // The clock and process-local sequence still provide collision retry input.
  }
  std::ostringstream out;
  out << std::hex << token;
  return out.str();
}

static bool replace_output_file(const std::filesystem::path& temporary_path,
                                const std::filesystem::path& output_path) {
#ifdef _WIN32
  return MoveFileExW(temporary_path.c_str(),
                     output_path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  std::error_code ec;
  std::filesystem::rename(temporary_path, output_path, ec);
  return !ec;
#endif
}

static bool create_private_directory(const std::filesystem::path& path,
                                     std::error_code& ec) {
#ifdef _WIN32
  return std::filesystem::create_directory(path, ec);
#else
  if (::mkdir(path.c_str(), S_IRWXU) == 0) {
    if (::chmod(path.c_str(), S_IRWXU) != 0) {
      ec = std::error_code(errno, std::generic_category());
      std::error_code cleanup_ec;
      std::filesystem::remove(path, cleanup_ec);
      return false;
    }
    ec.clear();
    return true;
  }
  ec = std::error_code(errno, std::generic_category());
  return false;
#endif
}

class StagedOutputFile {
public:
  explicit StagedOutputFile(const std::string& output_path)
      : output_path(output_path) {
    const std::filesystem::path parent =
        this->output_path.parent_path().empty()
            ? std::filesystem::path(".")
            : this->output_path.parent_path();
    if (this->output_path.filename().empty()) return;

    for (int attempt = 0; attempt < 128; ++attempt) {
      const auto candidate = parent / (".lac-tmp." + temporary_output_suffix());
      std::error_code ec;
      if (!create_private_directory(candidate, ec)) {
        if (!ec || ec == std::make_error_code(std::errc::file_exists)) continue;
        return;
      }

      this->temporary_directory = candidate;
      this->temporary_path = candidate / "output";
      return;
    }
  }

  ~StagedOutputFile() {
    this->cleanup();
  }

  bool is_ready() const {
    return !this->temporary_path.empty();
  }

  std::string path() const {
    return this->temporary_path.string();
  }

  bool publish(const std::string& input_path) {
    if (!this->is_ready()) return false;
    if (paths_refer_to_same_file(input_path, this->output_path.string())) return false;
    if (!replace_output_file(this->temporary_path, this->output_path)) return false;

    this->temporary_path.clear();
    std::error_code ec;
    std::filesystem::remove(this->temporary_directory, ec);
    if (!ec) this->temporary_directory.clear();
    return true;
  }

private:
  void cleanup() {
    std::error_code ec;
    if (!this->temporary_path.empty()) {
      std::filesystem::remove(this->temporary_path, ec);
    }
    ec.clear();
    if (!this->temporary_directory.empty()) {
      std::filesystem::remove(this->temporary_directory, ec);
    }
  }

  std::filesystem::path output_path;
  std::filesystem::path temporary_directory;
  std::filesystem::path temporary_path;
};

static bool parse_threads_flag(const std::string& flag, size_t& out_threads) {
  const std::string prefix = "--threads=";
  if (flag.rfind(prefix, 0) != 0) {
    return false;
  }

  const std::string value = flag.substr(prefix.size());
  if (value.empty()) {
    throw std::invalid_argument("--threads requires a positive integer");
  }
  for (char c : value) {
    if (c < '0' || c > '9') {
      throw std::invalid_argument("--threads requires a positive integer");
    }
  }

  size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size() || parsed == 0 ||
      parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    throw std::invalid_argument("--threads requires a positive integer");
  }
  out_threads = static_cast<size_t>(parsed);
  return true;
}

static void usage() {
  std::cerr << "Usage:\n";
  std::cerr << "  lac_cli encode input.wav output.lac [--stereo-mode=lr|ms] [--threads=N] [--debug-threads] [--debug-lpc] [--debug-stereo-est] [--debug-zr] [--debug-partitions] [--no-partitioning]\n";
  std::cerr << "  lac_cli decode input.lac output.wav [--threads=N] [--debug-threads]\n";
  std::cerr << "  lac_cli selftest\n";
}

int main(int argc, char** argv) {
  try {
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
      if (paths_refer_to_same_file(in_path, out_path)) {
        std::cerr << "Input and output paths must be different\n";
        return 1;
      }
      uint8_t stereo_mode = 2; // default to auto; allow override via flags
      bool debug_threads = false;
      bool debug_lpc = false;
      bool debug_stereo_est = false;
      bool debug_zr = false;
      bool debug_partitions = false;
      bool partitioning_enabled = true;
      size_t thread_count = 0;
      for (int i = 4; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--debug-threads") {
          debug_threads = true;
        } else if (flag == "--debug-stereo-est") {
          debug_stereo_est = true;
        } else if (flag == "--debug-lpc") {
          debug_lpc = true;
        } else if (flag == "--debug-zr") {
          debug_zr = true;
        } else if (flag == "--debug-partitions") {
          debug_partitions = true;
        } else if (flag == "--no-partitioning") {
          partitioning_enabled = false;
        } else if (flag == "--stereo-mode=lr") {
          stereo_mode = 0;
        } else if (flag == "--stereo-mode=ms") {
          stereo_mode = 1;
        } else if (parse_threads_flag(flag, thread_count)) {
          // Parsed above.
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
      uint8_t effective_stereo_mode = stereo_mode;
      if (channels == 1) {
        effective_stereo_mode = 0;
      } else if (channels == 2) {
        // If user did not specify, stereo_mode remains 2 (auto).
        // If user specified lr/ms, stereo_mode is already 0/1.
      }
      LAC::Encoder encoder(12, effective_stereo_mode, sample_rate, bit_depth, debug_lpc, debug_stereo_est, debug_zr);
      encoder.set_partitioning_enabled(partitioning_enabled);
      encoder.set_debug_partitions(debug_partitions);
      encoder.set_thread_count(thread_count);
      LAC::ThreadCollector collector;
      LAC::ThreadCollector* collector_ptr = (debug_threads ? &collector : nullptr);
      std::vector<uint8_t> bitstream = encoder.encode(left, right, collector_ptr);
      if (debug_zr) {
        LAC::Encoder baseline(12, effective_stereo_mode, sample_rate, bit_depth, debug_lpc, debug_stereo_est, false);
        baseline.set_zero_run_enabled(false);
        baseline.set_partitioning_enabled(partitioning_enabled);
        baseline.set_debug_partitions(debug_partitions);
        baseline.set_thread_count(thread_count);
        std::vector<uint8_t> baseline_bs = baseline.encode(left, right);
        double gain = 0.0;
        if (!baseline_bs.empty()) {
          gain = (1.0 - static_cast<double>(bitstream.size()) / static_cast<double>(baseline_bs.size())) * 100.0;
        }
        std::cout << "[debug-zr] baseline_bytes=" << baseline_bs.size()
          << " zr_bytes=" << bitstream.size()
          << " gain=" << gain << "%\n";
      }
      StagedOutputFile staged_output(out_path);
      if (!staged_output.is_ready() ||
          !save_file(staged_output.path(), bitstream) ||
          !staged_output.publish(in_path)) {
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
      if (paths_refer_to_same_file(in_path, out_path)) {
        std::cerr << "Input and output paths must be different\n";
        return 1;
      }
      bool debug_threads = false;
      size_t thread_count = 0;
      for (int i = 4; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--debug-threads") {
          debug_threads = true;
        } else if (parse_threads_flag(flag, thread_count)) {
          // Parsed above.
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
      decoder.set_thread_count(thread_count);
      std::vector<int32_t> left, right;
      FrameHeader hdr;
      decoder.decode(bitstream.data(), bitstream.size(), left, right, &hdr);
      if (left.empty()) {
        std::cerr << "Decode failed or produced no samples\n";
        return 1;
      }
      StagedOutputFile staged_output(out_path);
      if (!staged_output.is_ready() ||
          !write_wav_unchecked_samples(staged_output.path(), left, right, hdr.channels, hdr.sample_rate, hdr.bit_depth) ||
          !staged_output.publish(in_path)) {
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

        LAC::Encoder enc_lr(12, 0, sample_rate, bit_depth);
        std::vector<uint8_t> bs_lr = enc_lr.encode(src_left, src_right);
        LAC::Decoder decoder;
        std::vector<int32_t> dec_lr_left, dec_lr_right;
        FrameHeader hdr_lr;
        auto t0 = std::chrono::high_resolution_clock::now();
        decoder.decode(bs_lr.data(), bs_lr.size(), dec_lr_left, dec_lr_right, &hdr_lr);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (dec_lr_left != src_left || dec_lr_right != src_right) {
          std::cerr << "LR roundtrip mismatch for sr=" << sample_rate << " depth=" << int(bit_depth) << "\n";
          return false;
        }
        if (hdr_lr.sample_rate != sample_rate || hdr_lr.bit_depth != bit_depth) {
          std::cerr << "LR header mismatch sr=" << hdr_lr.sample_rate << " depth=" << int(hdr_lr.bit_depth) << "\n";
          return false;
        }
        LAC::Encoder enc_ms(12, 1, sample_rate, bit_depth);
        std::vector<uint8_t> bs_ms = enc_ms.encode(src_left, src_right);
        std::vector<int32_t> dec_ms_left, dec_ms_right;
        FrameHeader hdr_ms;
        auto t2 = std::chrono::high_resolution_clock::now();
        decoder.decode(bs_ms.data(), bs_ms.size(), dec_ms_left, dec_ms_right, &hdr_ms);
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

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
