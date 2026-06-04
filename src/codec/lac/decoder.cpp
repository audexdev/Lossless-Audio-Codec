#include "decoder.hpp"
#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include "codec/block/decoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/simd/neon.hpp"

namespace LAC {
namespace {

constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz
constexpr uint64_t MAX_DECODED_PCM_BYTES = 1ULL << 30; // decoded int32 channel vectors
constexpr uint32_t MAX_BLOCK_COUNT =
    static_cast<uint32_t>(
        (MAX_DECODED_PCM_BYTES / sizeof(int32_t) +
         Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE - 1u) /
        Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE);

[[noreturn]] void throw_decode_error(const std::string& reason) {
  throw std::runtime_error(std::string("[decode-error] ") + reason);
}

[[noreturn]] void throw_decode_error(uint32_t block_index, const char* channel) {
  throw std::runtime_error(
      std::string("[decode-error] block=") + std::to_string(block_index) + " channel=" + channel);
}

bool is_valid_sample_for_depth(int64_t sample, uint8_t bit_depth) {
  if (bit_depth == 16) return sample >= -32768 && sample <= 32767;
  if (bit_depth == 24) return sample >= -0x800000 && sample <= 0x7FFFFF;
  return false;
}

bool validate_pcm_range(const int32_t* samples, size_t n, uint8_t bit_depth) {
  if (!samples) return false;
  for (size_t i = 0; i < n; ++i) {
    if (!is_valid_sample_for_depth(samples[i], bit_depth)) return false;
  }
  return true;
}

bool reconstruct_mid_side_in_place(int32_t* left_out,
                                   int32_t* right_out,
                                   size_t n,
                                   uint8_t bit_depth) {
  if (!left_out || !right_out) return false;
  for (size_t i = 0; i < n; ++i) {
    const int64_t m = left_out[i];
    const int64_t s = right_out[i];
    const int64_t l = m + ((s + (s & 1)) >> 1);
    const int64_t r = l - s;
    if (!is_valid_sample_for_depth(l, bit_depth) || !is_valid_sample_for_depth(r, bit_depth)) {
      return false;
    }
    left_out[i] = static_cast<int32_t>(l);
    right_out[i] = static_cast<int32_t>(r);
  }
  return true;
}

} // namespace

Decoder::Decoder(ThreadCollector* collector)
    : collector(collector), thread_count(0) {}

void Decoder::set_thread_count(size_t max_threads) {
  this->thread_count = max_threads;
}

void Decoder::decode(const uint8_t* data,
                     size_t size,
                     std::vector<int32_t>& left,
                     std::vector<int32_t>& right,
                     FrameHeader* out_header) {
  left.clear();
  right.clear();

  if (data == nullptr || size == 0) {
    throw_decode_error("empty input");
  }

  FrameHeader hdr;
  size_t header_bytes = 0;
  if (!FrameHeader::parse(data, size, hdr, header_bytes)) {
    throw_decode_error("invalid frame header");
  }

  if (size < header_bytes) throw_decode_error("truncated frame payload");
  const uint8_t* payload = data + header_bytes;
  size_t payload_bytes = size - header_bytes;
  BitReader br(payload, payload_bytes);

  uint32_t block_count = br.read_bits(32);
  if (br.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) {
    throw_decode_error("invalid block count");
  }
  const bool has_block_payload_sizes = (hdr.version >= 3);
  const uint32_t table_words_per_block = has_block_payload_sizes ? 2u : 1u;
  if (block_count > br.bits_remaining() / (32u * table_words_per_block)) {
    throw_decode_error("truncated block size table");
  }

  std::vector<uint32_t> block_sizes(block_count);
  std::vector<uint32_t> block_payload_sizes;
  if (has_block_payload_sizes) {
    block_payload_sizes.resize(block_count);
  }
  uint64_t total_samples = 0;
  uint64_t total_block_payload_bytes = 0;
  for (uint32_t i = 0; i < block_count; ++i) {
    uint32_t sz = br.read_bits(32);
    if (br.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE ||
        (i + 1u < block_count && sz < Block::MIN_CANONICAL_NON_FINAL_BLOCK_SIZE)) {
      throw_decode_error("invalid block size");
    }
    total_samples += sz;
    if (total_samples > MAX_TOTAL_SAMPLES) {
      throw_decode_error("total samples exceed maximum");
    }
    block_sizes[i] = sz;
    if (has_block_payload_sizes) {
      const uint32_t payload_size = br.read_bits(32);
      if (br.has_error() || payload_size == 0) {
        throw_decode_error("invalid compressed block size");
      }
      total_block_payload_bytes += payload_size;
      if (total_block_payload_bytes > payload_bytes) {
        throw_decode_error("compressed block sizes exceed frame payload");
      }
      block_payload_sizes[i] = payload_size;
    }
  }
  const uint64_t decoded_pcm_bytes =
      total_samples * static_cast<uint64_t>(hdr.channels) * sizeof(int32_t);
  if (decoded_pcm_bytes > MAX_DECODED_PCM_BYTES) {
    throw_decode_error("decoded PCM allocation exceeds maximum");
  }
  const uint64_t wav_data_bytes =
      total_samples * static_cast<uint64_t>(hdr.channels) * (hdr.bit_depth / 8u);
  if (36u + wav_data_bytes + (wav_data_bytes & 1u) > std::numeric_limits<uint32_t>::max()) {
    throw_decode_error("decoded WAV data exceeds RIFF limit");
  }

  const bool isStereo = (hdr.channels == 2);
  const bool perBlockStereo = isStereo && (hdr.stereo_mode == 2);
  const bool forceMidSide = isStereo && (hdr.stereo_mode == 1);

  std::vector<size_t> blockOffsets(block_count);
  size_t running = 0;
  for (uint32_t i = 0; i < block_count; ++i) {
    blockOffsets[i] = running;
    running += block_sizes[i];
  }

  std::vector<int32_t> decoded_left(running, 0);
  std::vector<int32_t> decoded_right;
  if (isStereo) {
    decoded_right.assign(running, 0);
  }

  auto decode_block = [&](uint32_t i, BitReader& block_reader) {
    bool mid_side = false;
    if (perBlockStereo) {
      uint32_t mode_flag = block_reader.read_bits(8);
      if (block_reader.has_error() || mode_flag > 1u) {
        throw_decode_error("invalid per-block stereo flag");
      }
      mid_side = (mode_flag == 1u);
    } else if (forceMidSide) {
      mid_side = true;
    }

    Block::Decoder blockDec;
    size_t offset = blockOffsets[i];
    int32_t* left_out = decoded_left.data() + offset;
    int32_t* right_out = isStereo ? decoded_right.data() + offset : nullptr;
    if (!blockDec.decode_into(block_reader, block_sizes[i], left_out)) {
      throw_decode_error(i, "primary");
    }

    if (isStereo) {
      if (!blockDec.decode_into(block_reader, block_sizes[i], right_out)) {
        throw_decode_error(i, "secondary");
      }
    }

    if (!isStereo) {
      if (!validate_pcm_range(left_out, block_sizes[i], hdr.bit_depth)) {
        throw_decode_error("decoded sample outside PCM bit depth");
      }
    } else if (mid_side) {
      if (!reconstruct_mid_side_in_place(left_out, right_out, block_sizes[i], hdr.bit_depth)) {
        throw_decode_error("decoded sample outside PCM bit depth");
      }
    } else {
      if (!validate_pcm_range(left_out, block_sizes[i], hdr.bit_depth) ||
          !validate_pcm_range(right_out, block_sizes[i], hdr.bit_depth)) {
        throw_decode_error("decoded sample outside PCM bit depth");
      }
    }
  };

  if (!has_block_payload_sizes) {
    if (this->collector) {
      this->collector->record(std::this_thread::get_id());
    }
    for (uint32_t i = 0; i < block_count; ++i) {
      decode_block(i, br);
    }
    if (br.bits_remaining() != 0u) {
      throw_decode_error("trailing frame payload");
    }
  } else {
    if ((br.bits_remaining() & 7u) != 0u) {
      throw_decode_error("unaligned compressed block payload");
    }
    const size_t available_payload_bytes = br.bits_remaining() / 8u;
    if (total_block_payload_bytes != available_payload_bytes) {
      throw_decode_error("compressed block sizes do not match frame payload");
    }

    const uint8_t* block_payload = payload + (payload_bytes - available_payload_bytes);
    std::vector<size_t> block_payload_offsets(block_count);
    size_t payload_offset = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
      block_payload_offsets[i] = payload_offset;
      payload_offset += block_payload_sizes[i];
    }

    size_t hardware_threads =
        std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t thread_limit = this->thread_count;  // 0 = auto; env is resolved by the CLI
    if (thread_limit > 0) {
      hardware_threads = std::min(hardware_threads, thread_limit);
    }
    const size_t worker_count = std::min<size_t>(hardware_threads, block_count);
    std::atomic<uint32_t> next_block{0};
    std::atomic<bool> stop_requested{false};
    std::mutex error_mutex;
    std::exception_ptr worker_error;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    // Join every started worker even if thread construction throws mid-startup,
    // so a partial launch cannot destroy joinable threads and call std::terminate.
    struct WorkerJoinGuard {
      std::vector<std::thread>& workers;
      ~WorkerJoinGuard() {
        for (auto& worker : workers) {
          if (worker.joinable()) worker.join();
        }
      }
    } worker_join_guard{workers};

    for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back([&]() {
        try {
          if (this->collector) {
            this->collector->record(std::this_thread::get_id());
          }
          while (!stop_requested.load(std::memory_order_acquire)) {
            const uint32_t block_idx = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block_idx >= block_count) return;
            BitReader block_reader(block_payload + block_payload_offsets[block_idx],
                                   block_payload_sizes[block_idx]);
            decode_block(block_idx, block_reader);
            if (block_reader.bits_remaining() != 0u) {
              throw_decode_error(block_idx, "trailing-payload");
            }
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
    if (worker_error) {
      std::rethrow_exception(worker_error);
    }
  }

  if (hdr.channels == 2 && decoded_right.size() != decoded_left.size()) {
    throw_decode_error("stereo channel size mismatch");
  }

  left.swap(decoded_left);
  right.swap(decoded_right);
  if (out_header) {
    *out_header = hdr;
  }
}

} // namespace LAC
