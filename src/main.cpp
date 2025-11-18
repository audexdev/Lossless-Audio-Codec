#include <iostream>
#include <vector>
#include <cmath>

#include "codec/lpc/lpc.hpp"

int main() {
    std::vector<int32_t> pcm;
    pcm.reserve(512);

    const double pi = 3.14159265358979323846;
    for (int i = 0; i < 512; ++i) {
        double v = std::sin(2.0 * pi * static_cast<double>(i) / 64.0);
        pcm.push_back(static_cast<int32_t>(std::round(v * 30000.0)));
    }

    int order = 10;
    LPC lpc(order);

    std::vector<int16_t> coeffs_q15;
    lpc.analyze_block_q15(pcm, coeffs_q15);

    std::vector<int32_t> residual;
    lpc.compute_residual_q15(pcm, coeffs_q15, residual);

    std::vector<int32_t> restored;
    lpc.restore_from_residual_q15(residual, coeffs_q15, restored);

    bool ok = true;
    for (size_t i = 0; i < pcm.size(); ++i) {
        if (pcm[i] != restored[i]) {
            ok = false;
            std::cout << "Mismatch at " << i
                      << " expected=" << pcm[i]
                      << " got=" << restored[i] << "\n";
        }
    }

    if (ok) {
        std::cout << "LPC Q15 TEST PASSED\n";
    } else {
        std::cout << "LPC Q15 TEST FAILED\n";
    }

    return 0;
}