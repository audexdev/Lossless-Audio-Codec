#include "codec/simd/rice_neon.hpp"
#include "codec/simd/neon.hpp"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

inline void rice_unsigned_scalar(const int32_t* residual,
                                 uint32_t* u_values,
                                 size_t n) {
    if (!residual || !u_values) return;
    for (size_t i = 0; i < n; ++i) {
        int32_t r = residual[i];
        u_values[i] = static_cast<uint32_t>((r << 1) ^ (r >> 31));
    }
}

inline void rice_prefix_scalar(const uint32_t* u_values,
                               uint64_t* prefix_sums,
                               size_t n) {
    if (!u_values || !prefix_sums) return;
    uint64_t running = 0;
    for (size_t i = 0; i < n; ++i) {
        running += u_values[i];
        prefix_sums[i] = running;
    }
}

} // namespace

namespace SIMD {

void rice_unsigned_simd_or_scalar(const int32_t* residual,
                                  uint32_t* u_values,
                                  size_t n) {
    if (!residual || !u_values || n == 0) return;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (kHasNeon) {
        size_t i = 0;
        size_t vec = n & ~static_cast<size_t>(3);
        for (; i < vec; i += 4) {
            int32x4_t v = vld1q_s32(residual + i);
            int32x4_t left = vshlq_n_s32(v, 1);
            int32x4_t sign = vshrq_n_s32(v, 31);
            int32x4_t u = veorq_s32(left, sign);
            vst1q_u32(u_values + i, vreinterpretq_u32_s32(u));
        }
        for (; i < n; ++i) {
            int32_t r = residual[i];
            u_values[i] = static_cast<uint32_t>((r << 1) ^ (r >> 31));
        }
        return;
    }
#endif
    rice_unsigned_scalar(residual, u_values, n);
}

void rice_prefix_sum_simd_or_scalar(const uint32_t* u_values,
                                    uint64_t* prefix_sums,
                                    size_t n) {
    if (!u_values || !prefix_sums || n == 0) return;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (kHasNeon) {
        size_t i = 0;
        size_t vec = n & ~static_cast<size_t>(3);
        uint64_t running = 0;
        for (; i < vec; i += 4) {
            uint32x4_t v = vld1q_u32(u_values + i);
            uint64x2_t low = vmovl_u32(vget_low_u32(v));
            uint64x2_t high = vmovl_u32(vget_high_u32(v));

            uint64x2_t shifted_low = vextq_u64(vdupq_n_u64(0), low, 1);
            low = vaddq_u64(low, shifted_low);
            uint64x2_t base_low = vdupq_n_u64(running);
            low = vaddq_u64(low, base_low);
            vst1q_u64(prefix_sums + i, low);
            running = vgetq_lane_u64(low, 1);

            uint64x2_t shifted_high = vextq_u64(vdupq_n_u64(0), high, 1);
            high = vaddq_u64(high, shifted_high);
            uint64x2_t base_high = vdupq_n_u64(running);
            high = vaddq_u64(high, base_high);
            vst1q_u64(prefix_sums + i + 2, high);
            running = vgetq_lane_u64(high, 1);
        }
        for (; i < n; ++i) {
            running += u_values[i];
            prefix_sums[i] = running;
        }
        return;
    }
#endif
    rice_prefix_scalar(u_values, prefix_sums, n);
}

} // namespace SIMD
