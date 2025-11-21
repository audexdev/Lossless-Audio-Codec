#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/block/encoder.hpp"
#include "codec/frame/frame_header.hpp"
#include "codec/lac/thread_collector.hpp"

namespace LAC {

class Encoder {
public:
    Encoder(uint8_t order,
            uint8_t stereo_mode = 0,
            uint32_t sample_rate = 44100,
            uint8_t bit_depth = 16,
            bool debug_lpc = false,
            bool debug_stereo_est = false,
            bool debug_zr = false);

    std::vector<uint8_t> encode(const std::vector<int32_t>& left,
                                const std::vector<int32_t>& right,
                                ThreadCollector* collector = nullptr);

    void set_zero_run_enabled(bool enabled);
    void set_partitioning_enabled(bool enabled);
    void set_debug_partitions(bool enabled);

private:
    uint8_t order;
    uint8_t stereo_mode;
    uint32_t sample_rate;
    uint8_t bit_depth;
    bool debug_lpc;
    bool debug_stereo_est;
    bool debug_zr;
    bool zero_run_enabled;
    bool partitioning_enabled;
    bool debug_partitions;
    std::vector<uint32_t> candidates;

    uint32_t select_block_size(const std::vector<int32_t>& left,
                               const std::vector<int32_t>& right,
                               size_t position) const;

    double block_complexity(const std::vector<int32_t>& channel,
                            size_t position,
                            size_t length) const;
};

} // namespace LAC
