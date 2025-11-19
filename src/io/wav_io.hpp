#pragma once
#include <cstdint>
#include <string>
#include <vector>

bool read_wav(const std::string& path,
              std::vector<int32_t>& left,
              std::vector<int32_t>& right,
              uint16_t& channels,
              uint32_t& sample_rate,
              uint8_t& bit_depth);

bool write_wav(const std::string& path,
               const std::vector<int32_t>& left,
               const std::vector<int32_t>& right,
               uint16_t channels,
               uint32_t sample_rate,
               uint8_t bit_depth);
