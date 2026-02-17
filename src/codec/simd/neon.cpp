#include "codec/simd/neon.hpp"
#include <algorithm>
#include <limits>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

constexpr int kResidualFallbackOrders[] = {12, 10, 8, 6, 4};

inline void ms_encode_scalar_impl(const int32_t* L, const int32_t* R,
                                  int32_t* M, int32_t* S,
                                  size_t n) {
    if (!L || !R || !M || !S || n == 0) return;
    for (size_t i = 0; i < n; ++i) {
        int32_t l = L[i];
        int32_t r = R[i];
        M[i] = (l + r) >> 1;
        S[i] = l - r;
    }
}

inline void copy_pcm_to_residual(const int32_t* pcm,
                                 int32_t* residual,
                                 size_t n) {
    if (!pcm || !residual) return;
    for (size_t i = 0; i < n; ++i) {
        residual[i] = pcm[i];
    }
}

inline void append_unique_order(std::vector<int>& out, int order) {
    if (std::find(out.begin(), out.end(), order) == out.end()) {
        out.push_back(order);
    }
}

inline std::vector<int> build_residual_attempt_orders(int start_order,
                                                      int max_order) {
    std::vector<int> attempts;
    start_order = std::max(0, std::min(start_order, max_order));
    append_unique_order(attempts, start_order);
    for (int cand : kResidualFallbackOrders) {
        if (cand < start_order && cand <= max_order) {
            append_unique_order(attempts, cand);
        }
    }
    append_unique_order(attempts, 0);
    return attempts;
}

inline void lpc_residual_scalar_unchecked_span(const int32_t* pcm,
                                               const int16_t* coeffs_q15,
                                               int32_t* residual,
                                               size_t start,
                                               size_t end,
                                               int order) {
    if (!pcm || !coeffs_q15 || !residual || start >= end) return;
    if (order <= 0) {
        for (size_t i = start; i < end; ++i) {
            residual[i] = pcm[i];
        }
        return;
    }

    for (size_t sample = start; sample < end; ++sample) {
        const int taps = std::min(order, static_cast<int>(sample));
        int64_t acc = 0;
        for (int i = 1; i <= taps; ++i) {
            acc += static_cast<int64_t>(coeffs_q15[i]) *
                   static_cast<int64_t>(pcm[sample - static_cast<size_t>(i)]);
        }
        const int64_t pred = (acc >> 15);
        const int64_t diff = static_cast<int64_t>(pcm[sample]) - pred;
        residual[sample] = static_cast<int32_t>(diff);
    }
}

inline bool lpc_residual_scalar_check_range(const int32_t* pcm,
                                            const int16_t* coeffs_q15,
                                            size_t n,
                                            int order) {
    if (!pcm || !coeffs_q15) return false;
    if (order <= 0) return true;

    const int64_t lo = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    const int64_t hi = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

    for (size_t sample = 0; sample < n; ++sample) {
        const int taps = std::min(order, static_cast<int>(sample));
        int64_t acc = 0;
        for (int i = 1; i <= taps; ++i) {
            acc += static_cast<int64_t>(coeffs_q15[i]) *
                   static_cast<int64_t>(pcm[sample - static_cast<size_t>(i)]);
        }
        const int64_t pred = (acc >> 15);
        const int64_t diff = static_cast<int64_t>(pcm[sample]) - pred;
        if (diff < lo || diff > hi) {
            return false;
        }
    }
    return true;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline void neon_lpc_residual_safe_order(const int32_t* pcm,
                                         const int16_t* coeffs_q15,
                                         int32_t* residual,
                                         size_t n,
                                         int order) {
    if (!pcm || !coeffs_q15 || !residual || n == 0) return;
    if (order <= 0) {
        copy_pcm_to_residual(pcm, residual, n);
        return;
    }

    const size_t order_sz = static_cast<size_t>(order);
    const size_t warmup = (order_sz < n) ? order_sz : n;
    lpc_residual_scalar_unchecked_span(pcm, coeffs_q15, residual, 0, warmup, order);
    if (warmup >= n) return;

    size_t sample = warmup;
    std::vector<const int32_t*> history(order);
    for (int tap = 0; tap < order; ++tap) {
        history[tap] = pcm + warmup - static_cast<size_t>(tap + 1);
    }

    const size_t vec_samples = ((n - sample) / 4) * 4;
    const size_t vec_end = sample + vec_samples;

    for (; sample < vec_end; sample += 4) {
        int64x2_t acc_lo = vdupq_n_s64(0);
        int64x2_t acc_hi = vdupq_n_s64(0);

        int tap = 0;
        for (; tap + 3 < order; tap += 4) {
            int32_t c0 = static_cast<int32_t>(coeffs_q15[tap + 1]);
            int32_t c1 = static_cast<int32_t>(coeffs_q15[tap + 2]);
            int32_t c2 = static_cast<int32_t>(coeffs_q15[tap + 3]);
            int32_t c3 = static_cast<int32_t>(coeffs_q15[tap + 4]);

            const int32_t* ptr0 = history[tap + 0];
            int32x4_t prev0 = vld1q_s32(ptr0);
            acc_lo = vmlal_s32(acc_lo, vget_low_s32(prev0), vdup_n_s32(c0));
            acc_hi = vmlal_s32(acc_hi, vget_high_s32(prev0), vdup_n_s32(c0));
            history[tap + 0] = ptr0 + 4;

            const int32_t* ptr1 = history[tap + 1];
            int32x4_t prev1 = vld1q_s32(ptr1);
            acc_lo = vmlal_s32(acc_lo, vget_low_s32(prev1), vdup_n_s32(c1));
            acc_hi = vmlal_s32(acc_hi, vget_high_s32(prev1), vdup_n_s32(c1));
            history[tap + 1] = ptr1 + 4;

            const int32_t* ptr2 = history[tap + 2];
            int32x4_t prev2 = vld1q_s32(ptr2);
            acc_lo = vmlal_s32(acc_lo, vget_low_s32(prev2), vdup_n_s32(c2));
            acc_hi = vmlal_s32(acc_hi, vget_high_s32(prev2), vdup_n_s32(c2));
            history[tap + 2] = ptr2 + 4;

            const int32_t* ptr3 = history[tap + 3];
            int32x4_t prev3 = vld1q_s32(ptr3);
            acc_lo = vmlal_s32(acc_lo, vget_low_s32(prev3), vdup_n_s32(c3));
            acc_hi = vmlal_s32(acc_hi, vget_high_s32(prev3), vdup_n_s32(c3));
            history[tap + 3] = ptr3 + 4;
        }

        for (; tap < order; ++tap) {
            const int32_t coeff = static_cast<int32_t>(coeffs_q15[tap + 1]);
            const int32_t* ptr = history[tap];
            int32x4_t prev = vld1q_s32(ptr);
            acc_lo = vmlal_s32(acc_lo, vget_low_s32(prev), vdup_n_s32(coeff));
            acc_hi = vmlal_s32(acc_hi, vget_high_s32(prev), vdup_n_s32(coeff));
            history[tap] = ptr + 4;
        }

        const int64x2_t pred_lo = vshrq_n_s64(acc_lo, 15);
        const int64x2_t pred_hi = vshrq_n_s64(acc_hi, 15);

        const int64_t pred0 = vgetq_lane_s64(pred_lo, 0);
        const int64_t pred1 = vgetq_lane_s64(pred_lo, 1);
        const int64_t pred2 = vgetq_lane_s64(pred_hi, 0);
        const int64_t pred3 = vgetq_lane_s64(pred_hi, 1);

        residual[sample + 0] = static_cast<int32_t>(static_cast<int64_t>(pcm[sample + 0]) - pred0);
        residual[sample + 1] = static_cast<int32_t>(static_cast<int64_t>(pcm[sample + 1]) - pred1);
        residual[sample + 2] = static_cast<int32_t>(static_cast<int64_t>(pcm[sample + 2]) - pred2);
        residual[sample + 3] = static_cast<int32_t>(static_cast<int64_t>(pcm[sample + 3]) - pred3);
    }

    if (vec_end < n) {
        lpc_residual_scalar_unchecked_span(pcm, coeffs_q15, residual, vec_end, n, order);
    }
}
#endif

inline void lpc_residual_with_fallback(const int32_t* pcm,
                                       const int16_t* coeffs_q15,
                                       int32_t* residual,
                                       size_t n,
                                       int requested_order,
                                       bool allow_neon,
                                       int* used_order_inout) {
    if (!pcm || !coeffs_q15 || !residual) return;
    if (n == 0) {
        if (used_order_inout) *used_order_inout = 0;
        return;
    }

    const int max_order = std::max(0, std::min(requested_order, static_cast<int>(n - 1)));
    int start_order = max_order;
    if (used_order_inout != nullptr) {
        start_order = std::max(0, std::min(*used_order_inout, max_order));
    }

    const std::vector<int> attempts = build_residual_attempt_orders(start_order, max_order);

    int final_order = 0;
    bool success = false;
    for (int order : attempts) {
        if (order <= 0) {
            copy_pcm_to_residual(pcm, residual, n);
            final_order = 0;
            success = true;
            break;
        }

        const bool safe = lpc_residual_scalar_check_range(pcm, coeffs_q15, n, order);
        if (!safe) {
            continue;
        }

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (allow_neon) {
            neon_lpc_residual_safe_order(pcm, coeffs_q15, residual, n, order);
            final_order = order;
            success = true;
            break;
        }
#endif

        lpc_residual_scalar_unchecked_span(pcm, coeffs_q15, residual, 0, n, order);
        final_order = order;
        success = true;
        break;
    }

    if (!success) {
        copy_pcm_to_residual(pcm, residual, n);
        final_order = 0;
    }

    if (used_order_inout != nullptr) {
        *used_order_inout = final_order;
    }
}

} // namespace

namespace SIMD {

bool neon_available() {
    return kHasNeon;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
int64_t neon_absdiff_sum(const int32_t* pcm, size_t n) {
    if (!pcm || n <= 1) return 0;

    const size_t diff_count = n - 1;
    size_t i = 0;
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    const size_t vec_iters = diff_count / 4;

    for (size_t chunk = 0; chunk < vec_iters; ++chunk, i += 4) {
        int32x4_t curr = vld1q_s32(pcm + i);
        int32x4_t next = vld1q_s32(pcm + i + 1);
        int32x4_t diff = vabdq_s32(next, curr);
        acc0 = vaddq_s64(acc0, vmovl_s32(vget_low_s32(diff)));
        acc1 = vaddq_s64(acc1, vmovl_s32(vget_high_s32(diff)));
    }

    int64x2_t acc = vaddq_s64(acc0, acc1);
    int64_t sum = vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);
    const size_t processed = i;

    if (processed < diff_count) {
        int32_t prev = pcm[processed];
        for (size_t idx = processed + 1; idx < n; ++idx) {
            int32_t cur = pcm[idx];
            int64_t d = static_cast<int64_t>(cur) - static_cast<int64_t>(prev);
            sum += (d >= 0 ? d : -d);
            prev = cur;
        }
    }

    return sum;
}

void neon_ms_encode(const int32_t* L, const int32_t* R,
                    int32_t* M, int32_t* S,
                    size_t n) {
    if (!L || !R || !M || !S || n == 0) return;

    size_t i = 0;
    const size_t limit = n & ~static_cast<size_t>(3);

    for (; i < limit; i += 4) {
        int32x4_t l = vld1q_s32(L + i);
        int32x4_t r = vld1q_s32(R + i);
        int32x4_t sum = vaddq_s32(l, r);
        int32x4_t mid = vshrq_n_s32(sum, 1);
        int32x4_t side = vsubq_s32(l, r);
        vst1q_s32(M + i, mid);
        vst1q_s32(S + i, side);
    }

    if (i < n) {
        ms_encode_scalar_impl(L + i, R + i, M + i, S + i, n - i);
    }
}
#else
int64_t neon_absdiff_sum(const int32_t*, size_t) {
    return 0;
}

void neon_ms_encode(const int32_t* L, const int32_t* R,
                    int32_t* M, int32_t* S,
                    size_t n) {
    ms_encode_scalar_impl(L, R, M, S, n);
}
#endif

void neon_lpc_residual(const int32_t* pcm,
                       const int16_t* coeffs_q15,
                       int32_t* residual,
                       size_t n,
                       int order,
                       int* used_order_inout) {
    lpc_residual_with_fallback(pcm,
                               coeffs_q15,
                               residual,
                               n,
                               order,
                               neon_available(),
                               used_order_inout);
}

void ms_encode_simd_or_scalar(const int32_t* L, const int32_t* R,
                              int32_t* M, int32_t* S,
                              size_t n) {
    if (neon_available()) {
        neon_ms_encode(L, R, M, S, n);
    } else {
        ms_encode_scalar_impl(L, R, M, S, n);
    }
}

void lpc_residual_simd_or_scalar(const int32_t* pcm,
                                 const int16_t* coeffs_q15,
                                 int32_t* residual,
                                 size_t n,
                                 int order,
                                 int* used_order_inout) {
    lpc_residual_with_fallback(pcm,
                               coeffs_q15,
                               residual,
                               n,
                               order,
                               neon_available(),
                               used_order_inout);
}

} // namespace SIMD
