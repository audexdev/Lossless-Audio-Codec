#include "decoder.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include "codec/block/decoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/simd/neon.hpp"

namespace LAC {
namespace {

constexpr uint32_t MAX_BLOCK_COUNT = 10'000'000;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz

[[noreturn]] void throw_decode_error(const std::string& reason) {
  throw std::runtime_error(std::string("[decode-error] ") + reason);
}

[[noreturn]] void throw_decode_error(uint32_t block_index, const char* channel) {
  throw std::runtime_error(
      std::string("[decode-error] block=") + std::to_string(block_index) + " channel=" + channel);
}

void reconstruct_mid_side_to_output(const std::vector<int32_t>& mid,
                                    const std::vector<int32_t>& side,
                                    int32_t* left_out,
                                    int32_t* right_out,
                                    size_t n) {
  if (!left_out || !right_out) return;
  for (size_t i = 0; i < n; ++i) {
    int32_t m = mid[i];
    int32_t s = side[i];
    int32_t l = m + ((s + (s & 1)) >> 1);
    left_out[i] = l;
    right_out[i] = l - s;
  }
}

} // namespace

Decoder::Decoder(ThreadCollector* collector)
    : collector(collector) {}

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

  std::vector<uint32_t> block_sizes(block_count);
  uint64_t total_samples = 0;
  for (uint32_t i = 0; i < block_count; ++i) {
    uint32_t sz = br.read_bits(32);
    if (br.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE) {
      throw_decode_error("invalid block size");
    }
    total_samples += sz;
    if (total_samples > MAX_TOTAL_SAMPLES) {
      throw_decode_error("total samples exceed maximum");
    }
    block_sizes[i] = sz;
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

  left.assign(running, 0);
  if (isStereo) {
    right.assign(running, 0);
  } else {
    right.clear();
  }

  if (this->collector) {
    this->collector->record(std::this_thread::get_id());
  }

  for (uint32_t i = 0; i < block_count; ++i) {
    bool mid_side = false;
    if (perBlockStereo) {
      uint32_t mode_flag = br.read_bits(8);
      if (br.has_error()) throw_decode_error("invalid per-block stereo flag");
      mid_side = (mode_flag != 0);
    } else if (forceMidSide) {
      mid_side = true;
    }

    Block::Decoder blockDec;
    std::vector<int32_t> primary_pcm;
    if (!blockDec.decode(br, block_sizes[i], primary_pcm)) {
      throw_decode_error(i, "primary");
    }

    std::vector<int32_t> secondary_pcm;
    if (isStereo) {
      if (!blockDec.decode(br, block_sizes[i], secondary_pcm)) {
        throw_decode_error(i, "secondary");
      }
    }

    size_t offset = blockOffsets[i];
    if (!isStereo) {
      std::copy(primary_pcm.begin(), primary_pcm.end(), left.begin() + offset);
    } else if (mid_side) {
      reconstruct_mid_side_to_output(primary_pcm,
                                     secondary_pcm,
                                     left.data() + offset,
                                     right.data() + offset,
                                     primary_pcm.size());
    } else {
      std::copy(primary_pcm.begin(), primary_pcm.end(), left.begin() + offset);
      std::copy(secondary_pcm.begin(), secondary_pcm.end(), right.begin() + offset);
    }
  }

  if (hdr.channels == 2 && right.size() != left.size()) {
    left.clear();
    right.clear();
    throw_decode_error("stereo channel size mismatch");
  }

  if (out_header) {
    *out_header = hdr;
  }
}

} // namespace LAC
