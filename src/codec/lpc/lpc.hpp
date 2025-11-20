#pragma once
#include <vector>
#include <cstdint>

class LPC {
public:
    explicit LPC(int order);

    int get_order() const;

    bool analyze_block_q15(const std::vector<int32_t>& block,
                           std::vector<int16_t>& coeffs_q15,
                           int& used_order,
                           long double* energy_out = nullptr) const;

    void compute_residual_q15(const std::vector<int32_t>& block,
                              const std::vector<int16_t>& coeffs_q15,
                              std::vector<int32_t>& residual) const;

    void restore_from_residual_q15(const std::vector<int32_t>& residual,
                                   const std::vector<int16_t>& coeffs_q15,
                                   std::vector<int32_t>& out_block) const;

private:
    int order;

    void autocorrelation(const std::vector<int32_t>& block,
                         std::vector<long double>& R) const;

    int levinson_durbin(const std::vector<long double>& R,
                        std::vector<long double>& a) const;

    int16_t quantize_coeff_q15(double c) const;
};
