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
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "codec/bitstream/bit_reader.hpp"
#include "codec/block/decoder.hpp"
#include "codec/lac/thread_limit.hpp"
#include "io/wav_io.hpp"
#include "codec/lac/encoder.hpp"
#include "codec/lac/decoder.hpp"

namespace {
constexpr uint64_t MAX_LAC_INPUT_BYTES = 1ULL << 30;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL;
constexpr uint64_t MAX_DECODED_PCM_BYTES = 1ULL << 30;
constexpr uint32_t MAX_BLOCK_COUNT =
    static_cast<uint32_t>(
        (MAX_DECODED_PCM_BYTES / sizeof(int32_t) +
         Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE - 1u) /
        Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE);
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

#ifndef _WIN32
enum class FastDecodeStatus {
  Unsupported,
  Ok,
  Failed
};

static bool cli_is_valid_sample_for_depth(int64_t sample, uint8_t bit_depth) {
  if (bit_depth == 16) return sample >= -32768 && sample <= 32767;
  if (bit_depth == 24) return sample >= -0x800000 && sample <= 0x7FFFFF;
  return false;
}

static bool cli_validate_pcm_range(const int32_t* samples, size_t n, uint8_t bit_depth) {
  if (!samples) return false;
  for (size_t i = 0; i < n; ++i) {
    if (!cli_is_valid_sample_for_depth(samples[i], bit_depth)) return false;
  }
  return true;
}

static bool cli_reconstruct_mid_side_in_place(int32_t* left,
                                              int32_t* right,
                                              size_t n,
                                              uint8_t bit_depth) {
  if (!left || !right) return false;
  for (size_t i = 0; i < n; ++i) {
    const int64_t m = left[i];
    const int64_t s = right[i];
    const int64_t l = m + ((s + (s & 1)) >> 1);
    const int64_t r = l - s;
    if (!cli_is_valid_sample_for_depth(l, bit_depth) ||
        !cli_is_valid_sample_for_depth(r, bit_depth)) {
      return false;
    }
    left[i] = static_cast<int32_t>(l);
    right[i] = static_cast<int32_t>(r);
  }
  return true;
}

static void put_u16_le(uint8_t*& out, uint16_t value) {
  *out++ = static_cast<uint8_t>(value & 0xFFu);
  *out++ = static_cast<uint8_t>((value >> 8) & 0xFFu);
}

static void put_u32_le(uint8_t*& out, uint32_t value) {
  *out++ = static_cast<uint8_t>(value & 0xFFu);
  *out++ = static_cast<uint8_t>((value >> 8) & 0xFFu);
  *out++ = static_cast<uint8_t>((value >> 16) & 0xFFu);
  *out++ = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

static void write_wav_header_to_buffer(uint8_t* dst,
                                       uint16_t channels,
                                       uint32_t sample_rate,
                                       uint8_t bit_depth,
                                       uint32_t block_align,
                                       uint32_t data_size,
                                       uint32_t riff_size) {
  uint8_t* out = dst;
  *out++ = 'R'; *out++ = 'I'; *out++ = 'F'; *out++ = 'F';
  put_u32_le(out, riff_size);
  *out++ = 'W'; *out++ = 'A'; *out++ = 'V'; *out++ = 'E';
  *out++ = 'f'; *out++ = 'm'; *out++ = 't'; *out++ = ' ';
  put_u32_le(out, 16);
  put_u16_le(out, 1);
  put_u16_le(out, channels);
  put_u32_le(out, sample_rate);
  put_u32_le(out, sample_rate * block_align);
  put_u16_le(out, static_cast<uint16_t>(block_align));
  put_u16_le(out, bit_depth);
  *out++ = 'd'; *out++ = 'a'; *out++ = 't'; *out++ = 'a';
  put_u32_le(out, data_size);
}

static void pack_pcm_to_wav_bytes(uint8_t* dst,
                                  const int32_t* left,
                                  const int32_t* right,
                                  uint32_t samples,
                                  uint16_t channels,
                                  uint8_t bit_depth) {
  uint8_t* out = dst;
  if (bit_depth == 16) {
    for (uint32_t i = 0; i < samples; ++i) {
      const uint32_t l = static_cast<uint32_t>(left[i]);
      *out++ = static_cast<uint8_t>(l & 0xFFu);
      *out++ = static_cast<uint8_t>((l >> 8) & 0xFFu);
      if (channels == 2) {
        const uint32_t r = static_cast<uint32_t>(right[i]);
        *out++ = static_cast<uint8_t>(r & 0xFFu);
        *out++ = static_cast<uint8_t>((r >> 8) & 0xFFu);
      }
    }
  } else {
    for (uint32_t i = 0; i < samples; ++i) {
      const uint32_t l = static_cast<uint32_t>(left[i]);
      *out++ = static_cast<uint8_t>(l & 0xFFu);
      *out++ = static_cast<uint8_t>((l >> 8) & 0xFFu);
      *out++ = static_cast<uint8_t>((l >> 16) & 0xFFu);
      if (channels == 2) {
        const uint32_t r = static_cast<uint32_t>(right[i]);
        *out++ = static_cast<uint8_t>(r & 0xFFu);
        *out++ = static_cast<uint8_t>((r >> 8) & 0xFFu);
        *out++ = static_cast<uint8_t>((r >> 16) & 0xFFu);
      }
    }
  }
}

static FastDecodeStatus decode_lac_v3_to_mapped_wav(const uint8_t* data,
                                                    size_t size,
                                                    const std::string& output_path,
                                                    size_t thread_count,
                                                    LAC::ThreadCollector* collector,
                                                    size_t& out_samples_per_channel,
                                                    std::string& error) {
  auto fail = [&](const std::string& reason) {
    error = reason;
    return FastDecodeStatus::Failed;
  };

  if (data == nullptr || size == 0) return fail("empty input");

  FrameHeader hdr;
  size_t header_bytes = 0;
  if (!FrameHeader::parse(data, size, hdr, header_bytes)) {
    return fail("invalid frame header");
  }
  if (hdr.version != 3) {
    return FastDecodeStatus::Unsupported;
  }

  if (size < header_bytes) return fail("truncated frame payload");
  const uint8_t* payload = data + header_bytes;
  const size_t payload_bytes = size - header_bytes;
  BitReader br(payload, payload_bytes);

  const uint32_t block_count = br.read_bits(32);
  if (br.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) {
    return fail("invalid block count");
  }
  if (block_count > br.bits_remaining() / 64u) {
    return fail("truncated block size table");
  }

  std::vector<uint32_t> block_sizes(block_count);
  std::vector<uint32_t> block_payload_sizes(block_count);
  uint64_t total_samples = 0;
  uint64_t total_block_payload_bytes = 0;
  for (uint32_t i = 0; i < block_count; ++i) {
    const uint32_t block_size = br.read_bits(32);
    if (br.has_error() || block_size == 0 || block_size > Block::MAX_BLOCK_SIZE ||
        (i + 1u < block_count && block_size < Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE)) {
      return fail("invalid block size");
    }
    total_samples += block_size;
    if (total_samples > MAX_TOTAL_SAMPLES) {
      return fail("total samples exceed maximum");
    }
    block_sizes[i] = block_size;

    const uint32_t payload_size = br.read_bits(32);
    if (br.has_error() || payload_size == 0) {
      return fail("invalid compressed block size");
    }
    total_block_payload_bytes += payload_size;
    if (total_block_payload_bytes > payload_bytes) {
      return fail("compressed block sizes exceed frame payload");
    }
    block_payload_sizes[i] = payload_size;
  }

  const uint64_t decoded_pcm_bytes =
      total_samples * static_cast<uint64_t>(hdr.channels) * sizeof(int32_t);
  if (decoded_pcm_bytes > MAX_DECODED_PCM_BYTES) {
    return fail("decoded PCM allocation exceeds maximum");
  }

  const uint16_t bytes_per_sample = static_cast<uint16_t>(hdr.bit_depth / 8u);
  const uint32_t block_align = static_cast<uint32_t>(hdr.channels) * bytes_per_sample;
  const uint64_t data_size_u64 = total_samples * block_align;
  const uint64_t data_padding = data_size_u64 & 1u;
  const uint64_t riff_size_u64 = 36u + data_size_u64 + data_padding;
  const uint64_t file_size_u64 = riff_size_u64 + 8u;
  if (riff_size_u64 > std::numeric_limits<uint32_t>::max() ||
      file_size_u64 > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    return fail("decoded WAV data exceeds RIFF limit");
  }
  const uint32_t data_size = static_cast<uint32_t>(data_size_u64);
  const uint32_t riff_size = static_cast<uint32_t>(riff_size_u64);

  if ((br.bits_remaining() & 7u) != 0u) {
    return fail("unaligned compressed block payload");
  }
  const size_t available_payload_bytes = br.bits_remaining() / 8u;
  if (total_block_payload_bytes != available_payload_bytes) {
    return fail("compressed block sizes do not match frame payload");
  }

  std::vector<size_t> block_offsets(block_count);
  std::vector<size_t> block_payload_offsets(block_count);
  size_t sample_offset = 0;
  size_t payload_offset = 0;
  for (uint32_t i = 0; i < block_count; ++i) {
    block_offsets[i] = sample_offset;
    block_payload_offsets[i] = payload_offset;
    sample_offset += block_sizes[i];
    payload_offset += block_payload_sizes[i];
  }
  const uint8_t* block_payload = payload + (payload_bytes - available_payload_bytes);
  out_samples_per_channel = static_cast<size_t>(total_samples);

  const int fd = ::open(output_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) return fail("failed to create WAV output");
  auto close_fd = [&]() {
    if (fd >= 0) {
      (void)::close(fd);
    }
  };

  if (::ftruncate(fd, static_cast<off_t>(file_size_u64)) != 0) {
    close_fd();
    return fail("failed to size WAV output");
  }

  void* mapped = ::mmap(nullptr,
                        static_cast<size_t>(file_size_u64),
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED,
                        fd,
                        0);
  if (mapped == MAP_FAILED) {
    close_fd();
    return fail("failed to map WAV output");
  }
  uint8_t* wav = static_cast<uint8_t*>(mapped);
  write_wav_header_to_buffer(wav, hdr.channels, hdr.sample_rate, hdr.bit_depth, block_align, data_size, riff_size);

  const bool is_stereo = hdr.channels == 2;
  const bool per_block_stereo = is_stereo && hdr.stereo_mode == 2;
  const bool force_mid_side = is_stereo && hdr.stereo_mode == 1;

  size_t hardware_threads =
      std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
  const size_t resolved_limit = LAC::resolve_thread_limit(thread_count);
  if (resolved_limit > 0) {
    hardware_threads = std::min(hardware_threads, resolved_limit);
  }
  const size_t worker_count = std::min<size_t>(hardware_threads, block_count);

  std::atomic<uint32_t> next_block{0};
  std::atomic<bool> stop_requested{false};
  std::mutex error_mutex;
  std::exception_ptr worker_error;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&, worker_idx]() {
      (void)worker_idx;
      try {
        if (collector) {
          collector->record(std::this_thread::get_id());
        }
        Block::Decoder block_decoder;
        std::vector<int32_t> left_block(Block::MAX_BLOCK_SIZE);
        std::vector<int32_t> right_block(is_stereo ? Block::MAX_BLOCK_SIZE : 0u);
        while (!stop_requested.load(std::memory_order_acquire)) {
          const uint32_t block_idx = next_block.fetch_add(1, std::memory_order_relaxed);
          if (block_idx >= block_count) return;

          BitReader block_reader(block_payload + block_payload_offsets[block_idx],
                                 block_payload_sizes[block_idx]);
          bool mid_side = false;
          if (per_block_stereo) {
            const uint32_t mode_flag = block_reader.read_bits(8);
            if (block_reader.has_error() || mode_flag > 1u) {
              throw std::runtime_error("invalid per-block stereo flag");
            }
            mid_side = (mode_flag == 1u);
          } else if (force_mid_side) {
            mid_side = true;
          }

          const uint32_t block_size = block_sizes[block_idx];
          int32_t* left_out = left_block.data();
          int32_t* right_out = is_stereo ? right_block.data() : nullptr;
          if (!block_decoder.decode_into(block_reader, block_size, left_out)) {
            throw std::runtime_error("failed to decode primary block");
          }
          if (is_stereo && !block_decoder.decode_into(block_reader, block_size, right_out)) {
            throw std::runtime_error("failed to decode secondary block");
          }
          if (block_reader.bits_remaining() != 0u) {
            throw std::runtime_error("trailing block payload");
          }

          if (!is_stereo) {
            if (!cli_validate_pcm_range(left_out, block_size, hdr.bit_depth)) {
              throw std::runtime_error("decoded sample outside PCM bit depth");
            }
          } else if (mid_side) {
            if (!cli_reconstruct_mid_side_in_place(left_out, right_out, block_size, hdr.bit_depth)) {
              throw std::runtime_error("decoded sample outside PCM bit depth");
            }
          } else {
            if (!cli_validate_pcm_range(left_out, block_size, hdr.bit_depth) ||
                !cli_validate_pcm_range(right_out, block_size, hdr.bit_depth)) {
              throw std::runtime_error("decoded sample outside PCM bit depth");
            }
          }

          const uint64_t wav_offset =
              44u + static_cast<uint64_t>(block_offsets[block_idx]) * block_align;
          pack_pcm_to_wav_bytes(wav + wav_offset,
                                left_out,
                                right_out,
                                block_size,
                                hdr.channels,
                                hdr.bit_depth);
        }
      } catch (...) {
        stop_requested.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!worker_error) {
          worker_error = std::current_exception();
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  bool ok = true;
  if (worker_error) {
    try {
      std::rethrow_exception(worker_error);
    } catch (const std::exception& ex) {
      error = ex.what();
    } catch (...) {
      error = "unknown decode worker failure";
    }
    ok = false;
  }
  if (::munmap(mapped, static_cast<size_t>(file_size_u64)) != 0) {
    if (ok) error = "failed to unmap WAV output";
    ok = false;
  }
  if (::close(fd) != 0) {
    if (ok) error = "failed to close WAV output";
    ok = false;
  }

  return ok ? FastDecodeStatus::Ok : FastDecodeStatus::Failed;
}
#endif

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
      StagedOutputFile staged_output(out_path);
      if (!staged_output.is_ready()) {
        std::cerr << "Failed to write WAV: " << out_path << "\n";
        return 1;
      }

      bool wrote_wav = false;
      size_t samples_per_channel = 0;
#ifndef _WIN32
      std::string fast_decode_error;
      const FastDecodeStatus fast_status =
          decode_lac_v3_to_mapped_wav(bitstream.data(),
                                      bitstream.size(),
                                      staged_output.path(),
                                      thread_count,
                                      decoderCollectorPtr,
                                      samples_per_channel,
                                      fast_decode_error);
      if (fast_status == FastDecodeStatus::Ok) {
        wrote_wav = true;
      } else if (fast_status == FastDecodeStatus::Failed) {
        std::cerr << "Decode failed: " << fast_decode_error << "\n";
        return 1;
      }
#endif
      if (!wrote_wav) {
        LAC::Decoder decoder(decoderCollectorPtr);
        decoder.set_thread_count(thread_count);
        std::vector<int32_t> left, right;
        FrameHeader hdr;
        decoder.decode(bitstream.data(), bitstream.size(), left, right, &hdr);
        if (left.empty()) {
          std::cerr << "Decode failed or produced no samples\n";
          return 1;
        }
        if (!write_wav_unchecked_samples(staged_output.path(), left, right, hdr.channels, hdr.sample_rate, hdr.bit_depth)) {
          std::cerr << "Failed to write WAV: " << out_path << "\n";
          return 1;
        }
        samples_per_channel = left.size();
      }
      if (!staged_output.publish(in_path)) {
        std::cerr << "Failed to write WAV: " << out_path << "\n";
        return 1;
      }
      std::cout << "Decoded " << in_path << " -> " << out_path << " (" << samples_per_channel << " samples per channel)\n";
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
