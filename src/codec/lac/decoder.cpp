#include "decoder.hpp"
#include <algorithm>
#include <limits>
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

bool copy_pcm_to_output(const std::vector<int32_t>& source,
                        int32_t* output,
                        uint8_t bit_depth) {
  if (!output) return false;
  for (size_t i = 0; i < source.size(); ++i) {
    if (!is_valid_sample_for_depth(source[i], bit_depth)) return false;
    output[i] = source[i];
  }
  return true;
}

bool reconstruct_mid_side_to_output(const std::vector<int32_t>& mid,
                                    const std::vector<int32_t>& side,
                                    int32_t* left_out,
                                    int32_t* right_out,
                                    size_t n,
                                    uint8_t bit_depth) {
  if (!left_out || !right_out || mid.size() != n || side.size() != n) return false;
  for (size_t i = 0; i < n; ++i) {
    const int64_t m = mid[i];
    const int64_t s = side[i];
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
  if (block_count > br.bits_remaining() / 32u) {
    throw_decode_error("truncated block size table");
  }

  std::vector<uint32_t> block_sizes(block_count);
  uint64_t total_samples = 0;
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

  if (this->collector) {
    this->collector->record(std::this_thread::get_id());
  }

  for (uint32_t i = 0; i < block_count; ++i) {
    bool mid_side = false;
    if (perBlockStereo) {
      uint32_t mode_flag = br.read_bits(8);
      if (br.has_error() || mode_flag > 1u) throw_decode_error("invalid per-block stereo flag");
      mid_side = (mode_flag == 1u);
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
      if (!copy_pcm_to_output(primary_pcm, decoded_left.data() + offset, hdr.bit_depth)) {
        throw_decode_error("decoded sample outside PCM bit depth");
      }
    } else if (mid_side) {
      if (!reconstruct_mid_side_to_output(primary_pcm,
                                          secondary_pcm,
                                          decoded_left.data() + offset,
                                          decoded_right.data() + offset,
                                          primary_pcm.size(),
                                          hdr.bit_depth)) {
        throw_decode_error("decoded sample outside PCM bit depth");
      }
    } else {
      if (!copy_pcm_to_output(primary_pcm, decoded_left.data() + offset, hdr.bit_depth) ||
          !copy_pcm_to_output(secondary_pcm, decoded_right.data() + offset, hdr.bit_depth)) {
        throw_decode_error("decoded sample outside PCM bit depth");
      }
    }
  }

  if (br.bits_remaining() != 0u) {
    throw_decode_error("trailing frame payload");
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
