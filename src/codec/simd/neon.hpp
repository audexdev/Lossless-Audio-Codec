#pragma once
#include <cstddef>
#include <cstdint>

namespace SIMD {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
constexpr bool kHasNeon = true;
#else
constexpr bool kHasNeon = false;
#endif

int64_t neon_absdiff_sum(const int32_t* pcm, size_t n);
bool neon_available();

void neon_ms_encode(const int32_t* L, const int32_t* R,
                    int32_t* M, int32_t* S,
                    size_t n);

void ms_encode_simd_or_scalar(const int32_t* L, const int32_t* R,
                              int32_t* M, int32_t* S,
                              size_t n);

void neon_lpc_residual(const int32_t* pcm,
                       const int16_t* coeffs_q15,
                       int32_t* residual,
                       size_t n,
                       int order);

void lpc_residual_simd_or_scalar(const int32_t* pcm,
                                 const int16_t* coeffs_q15,
                                 int32_t* residual,
                                 size_t n,
                                 int order);

} // namespace SIMD
