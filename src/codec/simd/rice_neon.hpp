#pragma once
#include <cstddef>
#include <cstdint>

namespace SIMD {

void rice_unsigned_simd_or_scalar(const int32_t* residual,
                                  uint32_t* u_values,
                                  size_t n);

void rice_prefix_sum_simd_or_scalar(const uint32_t* u_values,
                                    uint64_t* prefix_sums,
                                    size_t n);

} // namespace SIMD

