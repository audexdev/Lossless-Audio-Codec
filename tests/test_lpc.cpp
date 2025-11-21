#include "codec/lpc/lpc.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

void run_zerorun_tests();
void run_partitioning_tests();

namespace {

constexpr int kCandidates[] = {4, 6, 8, 10, 12};

inline uint32_t unsigned_from_residual(int32_t r) {
    return static_cast<uint32_t>((r << 1) ^ (r >> 31));
}

uint64_t estimate_rice_bits(const std::vector<int32_t>& residual) {
    if (residual.empty()) return 0;
    uint64_t sum_u = 0;
    for (int32_t r : residual) {
        sum_u += unsigned_from_residual(r);
    }
    const uint64_t mean = (sum_u + (residual.size() >> 1)) / residual.size();
    uint32_t k = 0;
    while ((1u << k) < mean && k < 31u) ++k;

    uint64_t bits = 0;
    for (int32_t r : residual) {
        const uint32_t u = unsigned_from_residual(r);
        const uint32_t q = (k >= 31u) ? 0u : (u >> k);
        bits += static_cast<uint64_t>(q) + 1u + k;
    }
    return bits;
}

uint64_t energy_sum(const std::vector<int32_t>& data) {
    uint64_t e = 0;
    for (int32_t v : data) {
        e += static_cast<uint64_t>(std::abs(static_cast<int64_t>(v)));
    }
    return e;
}

struct OrderEval {
    int order = 0;
    uint64_t bits = 0;
    uint64_t residual_energy = 0;
};

std::vector<OrderEval> evaluate_orders(const std::vector<int32_t>& block,
                                       uint64_t& raw_energy) {
    std::vector<OrderEval> evals;
    raw_energy = energy_sum(block);
    const int max_order = (block.size() > 1)
        ? static_cast<int>(std::min<size_t>(block.size() - 1, 32))
        : 0;

    for (int order : kCandidates) {
        if (order > max_order) continue;
        LPC lpc(order);
        std::vector<int16_t> coeffs;
        int used_order = 0;
        long double energy = 0.0L;
        if (!lpc.analyze_block_q15(block, coeffs, used_order, &energy)) {
            continue;
        }
        if (used_order < 4 || used_order > 12) {
            continue;
        }

        std::vector<int32_t> residual(block.size());
        lpc.compute_residual_q15(block, coeffs, residual);
        OrderEval eval;
        eval.order = used_order;
        eval.bits = estimate_rice_bits(residual);
        eval.residual_energy = energy_sum(residual);
        evals.push_back(eval);
    }
    return evals;
}

std::vector<int32_t> make_noise(size_t n, int32_t amplitude) {
    std::vector<int32_t> out(n);
    uint32_t state = 1;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = static_cast<int32_t>(static_cast<int32_t>(state >> 9) % amplitude);
    }
    return out;
}

std::vector<int32_t> make_ramp(size_t n, int32_t amplitude) {
    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<int32_t>((static_cast<int64_t>(amplitude) * static_cast<int64_t>(i)) / static_cast<int64_t>(n));
    }
    return out;
}

std::vector<int32_t> make_tone(size_t n, double freq, uint32_t sample_rate, int32_t amplitude) {
    std::vector<int32_t> out(n);
    const double two_pi = 6.28318530717958647692;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(sample_rate);
        out[i] = static_cast<int32_t>(std::sin(two_pi * freq * t) * amplitude);
    }
    return out;
}

std::vector<int32_t> make_near_silence(size_t n) {
    std::vector<int32_t> out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        out[i] = (i % 7 == 0) ? 1 : 0;
    }
    return out;
}

void run_case(const std::string& name, const std::vector<int32_t>& block) {
    uint64_t raw_energy = 0;
    std::vector<OrderEval> evals = evaluate_orders(block, raw_energy);
    assert(!evals.empty() && "At least one LPC order should succeed");

    auto best_it = std::min_element(
        evals.begin(), evals.end(),
        [](const OrderEval& a, const OrderEval& b) { return a.bits < b.bits; });
    assert(best_it != evals.end());
    assert(best_it->order >= 4 && best_it->order <= 12);
    if (raw_energy > 0) {
        assert(best_it->residual_energy <= raw_energy);
    }

    auto baseline_it = std::find_if(
        evals.begin(), evals.end(),
        [](const OrderEval& e) { return e.order == 4; });
    assert(baseline_it != evals.end() && "Order 4 should always be evaluated");
    const uint64_t baseline_energy = baseline_it->residual_energy;
    const uint64_t tolerance = baseline_energy / 10 + 32; // ~10% slack

    for (const auto& eval : evals) {
        assert(eval.order >= 4 && eval.order <= 12);
        assert(eval.residual_energy >= 0);
        if (raw_energy > 0) {
            assert(eval.residual_energy <= raw_energy);
        }
        if (eval.order > 4) {
            assert(eval.residual_energy <= baseline_energy + tolerance);
        }
    }

    std::cout << "Test " << name << ": best_order=" << best_it->order
              << " bits=" << best_it->bits
              << " rawE=" << raw_energy
              << " resE=" << best_it->residual_energy << "\n";
}

} // namespace

int main() {
    const size_t N = 2048;
    run_case("white_noise", make_noise(N, 30000));
    run_case("ramp", make_ramp(N, 50000));
    run_case("tone", make_tone(N, 440.0, 48000, 40000));
    run_case("near_silence", make_near_silence(N));
    run_zerorun_tests();
    run_partitioning_tests();
    return 0;
}
