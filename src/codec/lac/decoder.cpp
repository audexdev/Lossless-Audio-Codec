#include "decoder.hpp"
#include <algorithm>
#include <iostream>
#include <thread>
#include "codec/block/decoder.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/simd/neon.hpp"

namespace LAC {
namespace {

constexpr uint32_t MAX_BLOCK_COUNT = 10'000'000;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz

void reconstruct_mid_side(std::vector<int32_t>& mid,
                          std::vector<int32_t>& side,
                          std::vector<int32_t>& left,
                          std::vector<int32_t>& right) {
  size_t n = mid.size();
  left.resize(n);
  right.resize(n);
  for (size_t i = 0; i < n; ++i) {
    int32_t m = mid[i];
    int32_t s = side[i];
    int32_t l = m + ((s + (s & 1)) >> 1);
    left[i] = l;
    right[i] = l - s;
  }
}

} // namespace

Decoder::Decoder(ThreadCollector* collector)
    : collector(collector) {}

bool Decoder::decode(const uint8_t* data,
                     size_t size,
                     std::vector<int32_t>& left,
                     std::vector<int32_t>& right,
                     FrameHeader* out_header) {
  left.clear();
  right.clear();

  FrameHeader hdr;
  size_t header_bytes = 0;
  if (!FrameHeader::parse(data, size, hdr, header_bytes)) {
    return false;
  }

  if (size < header_bytes) return false;
  const uint8_t* payload = data + header_bytes;
  size_t payload_bytes = size - header_bytes;
  BitReader br(payload, payload_bytes);

  uint32_t block_count = br.read_bits(32);
  if (br.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) {
    return false;
  }

  std::vector<uint32_t> block_sizes(block_count);
  uint64_t total_samples = 0;
  for (uint32_t i = 0; i < block_count; ++i) {
    uint32_t sz = br.read_bits(32);
    if (br.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE) return false;
    total_samples += sz;
    if (total_samples > MAX_TOTAL_SAMPLES) return false;
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
      if (br.has_error()) return false;
      mid_side = (mode_flag != 0);
    } else if (forceMidSide) {
      mid_side = true;
    }

    Block::Decoder blockDec;
    std::vector<int32_t> primary_pcm;
    if (!blockDec.decode(br, block_sizes[i], primary_pcm)) {
      std::cerr << "[decode-error] block=" << i << " channel=primary\n";
      left.clear();
      right.clear();
      return false;
    }

    std::vector<int32_t> secondary_pcm;
    if (isStereo) {
      if (!blockDec.decode(br, block_sizes[i], secondary_pcm)) {
        std::cerr << "[decode-error] block=" << i << " channel=secondary\n";
        left.clear();
        right.clear();
        return false;
      }
    }

    size_t offset = blockOffsets[i];
    if (!isStereo) {
      std::copy(primary_pcm.begin(), primary_pcm.end(), left.begin() + offset);
    } else if (mid_side) {
      std::vector<int32_t> left_pcm, right_pcm;
      reconstruct_mid_side(primary_pcm, secondary_pcm, left_pcm, right_pcm);
      std::copy(left_pcm.begin(), left_pcm.end(), left.begin() + offset);
      std::copy(right_pcm.begin(), right_pcm.end(), right.begin() + offset);
    } else {
      std::copy(primary_pcm.begin(), primary_pcm.end(), left.begin() + offset);
      std::copy(secondary_pcm.begin(), secondary_pcm.end(), right.begin() + offset);
    }
  }

  if (hdr.channels == 2 && right.size() != left.size()) {
    left.clear();
    right.clear();
    return false;
  }

  if (out_header) {
    *out_header = hdr;
  }

  return true;
}

} // namespace LAC
