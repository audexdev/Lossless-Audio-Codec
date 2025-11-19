#pragma once
#include <vector>
#include <cstdint>
#include "codec/bitstream/bit_reader.hpp"
#include "codec/block/decoder.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/lac/thread_collector.hpp"

namespace LAC {

class Decoder {
public:
    explicit Decoder(ThreadCollector* collector = nullptr);

    bool decode(const uint8_t* data, size_t size, std::vector<int32_t>& left, std::vector<int32_t>& right, FrameHeader* out_header = nullptr);

private:
    ThreadCollector* collector;
};

} // namespace LAC
