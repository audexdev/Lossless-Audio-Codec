#pragma once
#include <cstdint>

namespace Block {

constexpr uint32_t MAX_BLOCK_SIZE = 16384;
#ifdef LAC_ENABLE_ZERORUN
constexpr uint32_t ZERO_RUN_MIN_LENGTH = 4;
constexpr uint32_t ZERO_RUN_LENGTH_K = 2;
#endif
constexpr uint32_t MIN_PARTITION_SIZE = 32;
constexpr uint8_t MAX_PARTITION_ORDER = 3;
constexpr uint8_t PARTITION_FLAG = 0x80;
constexpr uint8_t PARTITION_ORDER_SHIFT = 5;
constexpr uint8_t PARTITION_ORDER_MASK = 0x60;

}
