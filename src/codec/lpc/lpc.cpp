#include "lpc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
inline int32_t clamp_to_int32(int64_t value) {
    const int64_t lo = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    const int64_t hi = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if (value < lo) return std::numeric_limits<int32_t>::min();
    if (value > hi) return std::numeric_limits<int32_t>::max();
    return static_cast<int32_t>(value);
}
} // namespace

LPC::LPC(int order)
    : order(order)
{
}

int LPC::get_order() const {
    return this->order;
}

int16_t LPC::quantize_coeff_q15(double c) const {
    double scaled = std::round(c * 32768.0);
    if (scaled < -32768.0) scaled = -32768.0;
    if (scaled >  32767.0) scaled =  32767.0;
    return static_cast<int16_t>(scaled);
}

void LPC::autocorrelation(const std::vector<int32_t>& block,
                          std::vector<long double>& R) const {
    size_t N = block.size();
    if (N == 0) {
        std::fill(R.begin(), R.end(), 0.0L);
        return;
    }

    for (int k = 0; k <= this->order; ++k) {
        int64_t sum = 0;
        for (size_t n = k; n < N; ++n) {
            sum += static_cast<int64_t>(block[n]) *
                   static_cast<int64_t>(block[n - k]);
        }
        R[k] = static_cast<long double>(sum);
    }
}

int LPC::levinson_durbin(const std::vector<long double>& R,
                         std::vector<long double>& a) const {
    const long double eps = 1e-8L;
    std::vector<long double> E(this->order + 1);
    std::vector<long double> K(this->order + 1);
    std::vector<long double> prevA(this->order + 1);

    E[0] = R[0];
    if (!std::isfinite(E[0]) || E[0] < eps) {
        std::fill(a.begin(), a.end(), 0.0L);
        return 0;
    }

    int achieved_order = 0;

    for (int i = 1; i <= this->order; ++i) {
        long double acc = 0.0L;
        for (int j = 1; j < i; ++j) {
            acc += prevA[j] * R[i - j];
        }

        long double denom = E[i - 1];
        if (!std::isfinite(denom) || denom < eps) {
            break;
        }

        long double ki = (R[i] - acc) / denom;
        if (!std::isfinite(ki)) {
            break;
        }

        if (ki > 0.999L) ki = 0.999L;
        if (ki < -0.999L) ki = -0.999L;

        K[i] = ki;
        a[i] = ki;

        for (int j = 1; j < i; ++j) {
            a[j] = prevA[j] - K[i] * prevA[i - j];
        }

        for (int j = 1; j <= i; ++j) {
            prevA[j] = a[j];
        }

        E[i] = (1.0L - K[i] * K[i]) * E[i - 1];
        if (!std::isfinite(E[i]) || E[i] < eps) {
            achieved_order = i - 1;
            break;
        }
        achieved_order = i;
    }

    return achieved_order;
}

bool LPC::analyze_block_q15(const std::vector<int32_t>& block,
                            std::vector<int16_t>& coeffs_q15,
                            int& used_order,
                            long double* energy_out) const {
    std::vector<long double> R(this->order + 1);
    std::vector<long double> a(this->order + 1);

    this->autocorrelation(block, R);
    long double energy = (R.empty() ? 0.0L : R[0]);
    if (energy_out) {
        *energy_out = energy;
    }

    const long double min_energy = 1.0L;
    if (energy < min_energy) {
        R[0] = min_energy;
    }

    used_order = this->levinson_durbin(R, a);

    coeffs_q15.resize(this->order + 1);
    coeffs_q15[0] = 0;
    for (int i = 1; i <= used_order; ++i) {
        coeffs_q15[i] = this->quantize_coeff_q15(static_cast<double>(a[i]));
    }
    for (int i = used_order + 1; i <= this->order; ++i) {
        coeffs_q15[i] = 0;
    }
    return used_order > 0;

}

void LPC::compute_residual_q15(const std::vector<int32_t>& block,
                               const std::vector<int16_t>& coeffs_q15,
                               std::vector<int32_t>& residual) const {
    size_t N = block.size();
    residual.resize(N);

    std::vector<int32_t> recon(N);

    for (size_t n = 0; n < N; ++n) {
        int64_t acc = 0;
        for (int i = 1; i <= this->order; ++i) {
            if (n >= static_cast<size_t>(i)) {
                acc += static_cast<int64_t>(coeffs_q15[i]) *
                       static_cast<int64_t>(recon[n - i]);
            }
        }
        const int64_t pred = (acc >> 15);
        const int64_t sample = static_cast<int64_t>(block[n]) - pred;
        residual[n] = clamp_to_int32(sample);
        recon[n] = block[n];
    }
}

void LPC::restore_from_residual_q15(const std::vector<int32_t>& residual,
                                    const std::vector<int16_t>& coeffs_q15,
                                    std::vector<int32_t>& out_block) const {
    size_t N = residual.size();
    out_block.resize(N);

    for (size_t n = 0; n < N; ++n) {
        int64_t acc = 0;
        for (int i = 1; i <= this->order; ++i) {
            if (n >= static_cast<size_t>(i)) {
                acc += static_cast<int64_t>(coeffs_q15[i]) *
                       static_cast<int64_t>(out_block[n - i]);
            }
        }
        const int64_t pred = (acc >> 15);
        const int64_t sample = pred + static_cast<int64_t>(residual[n]);
        out_block[n] = clamp_to_int32(sample);
    }
}
