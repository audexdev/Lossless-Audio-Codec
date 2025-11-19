#include "lpc.hpp"
#include <cmath>

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
                          std::vector<double>& R) const {
    size_t N = block.size();
    for (int k = 0; k <= this->order; ++k) {
        double sum = 0.0;
        for (size_t n = k; n < N; ++n) {
            sum += static_cast<double>(block[n]) *
                   static_cast<double>(block[n - k]);
        }
        R[k] = sum;
    }
}

void LPC::levinson_durbin(const std::vector<double>& R,
                          std::vector<double>& a) const {
    std::vector<double> E(this->order + 1);
    std::vector<double> K(this->order + 1);
    std::vector<double> prevA(this->order + 1);

    E[0] = R[0];

    for (int i = 1; i <= this->order; ++i) {
        double acc = 0.0;
        for (int j = 1; j < i; ++j) {
            acc += prevA[j] * R[i - j];
        }

        double denom = E[i - 1];
        if (denom == 0.0) {
            K[i] = 0.0;
        } else {
            K[i] = (R[i] - acc) / (denom + 1e-12);
        }

        a[i] = K[i];

        for (int j = 1; j < i; ++j) {
            a[j] = prevA[j] - K[i] * prevA[i - j];
        }

        for (int j = 1; j <= i; ++j) {
            prevA[j] = a[j];
        }

        E[i] = (1.0 - K[i] * K[i]) * E[i - 1];
    }
}

void LPC::analyze_block_q15(const std::vector<int32_t>& block,
                            std::vector<int16_t>& coeffs_q15) const {
    std::vector<double> R(this->order + 1);
    std::vector<double> a(this->order + 1);

    this->autocorrelation(block, R);
    this->levinson_durbin(R, a);

    coeffs_q15.resize(this->order + 1);
    coeffs_q15[0] = 0;
    for (int i = 1; i <= this->order; ++i) {
        coeffs_q15[i] = this->quantize_coeff_q15(a[i]);
    }
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
        int32_t pred = static_cast<int32_t>(acc >> 15);
        residual[n] = block[n] - pred;
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
        int32_t pred = static_cast<int32_t>(acc >> 15);
        out_block[n] = pred + residual[n];
    }
}