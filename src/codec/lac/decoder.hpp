#pragma once
#include <vector>
#include <cstdint>
#include "codec/block/constants.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/lac/thread_collector.hpp"

namespace LAC {

class Decoder {
public:
    explicit Decoder(ThreadCollector* collector = nullptr);

    void decode(const uint8_t* data,
                size_t size,
                std::vector<int32_t>& left,
                std::vector<int32_t>& right,
                FrameHeader* out_header = nullptr);

private:
    ThreadCollector* collector;
};

} // namespace LAC
