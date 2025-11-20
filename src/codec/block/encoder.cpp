#include "encoder.hpp"
#include "codec/lpc/lpc.hpp"
#include "codec/rice/rice.hpp"
#include "codec/bitstream/bit_writer.hpp"
#include "codec/simd/neon.hpp"
#include "codec/simd/rice_neon.hpp"
#include <algorithm>
#include <limits>
#include <iostream>

namespace Block {

Encoder::Encoder(int order, bool debug_lpc) : order(order), debug_lpc(debug_lpc) {}

namespace {
constexpr int kOrderCandidates[] = {4, 6, 8, 10, 12};
constexpr size_t kInitialScanCount = 256;
constexpr uint32_t kInitialMaxK = 12;

inline uint32_t unsigned_from_residual(int32_t r) {
    return static_cast<uint32_t>((r << 1) ^ (r >> 31));
}

uint32_t estimate_initial_k(const std::vector<int32_t>& residual) {
    if (residual.empty()) return 0;
    const size_t count = std::min(kInitialScanCount, residual.size());

    uint64_t candidate_costs[kInitialMaxK + 1] = {0};
    uint64_t sum_abs = 0;

    for (size_t i = 0; i < count; ++i) {
        const uint32_t u = unsigned_from_residual(residual[i]);
        sum_abs += u;
        for (uint32_t k = 0; k <= kInitialMaxK; ++k) {
            const uint32_t q = (k == 31) ? 0u : (u >> k);
            // Rough bit cost: unary (q + 1) + remainder k bits.
            candidate_costs[k] += static_cast<uint64_t>(q) + 1u + k;
        }
    }

    // Fallback using mean if something unexpected occurs.
    uint32_t mean_based = 0;
    if (count > 0) {
        const uint64_t mean = (sum_abs + (count >> 1)) / count;
        while ((1u << mean_based) < mean && mean_based < 15u) {
            ++mean_based;
        }
    }

    uint32_t best_k = mean_based;
    uint64_t best_cost = std::numeric_limits<uint64_t>::max();
    for (uint32_t k = 0; k <= kInitialMaxK; ++k) {
        if (candidate_costs[k] < best_cost) {
            best_cost = candidate_costs[k];
            best_k = k;
        }
    }

    // Keep k conservative; cap at 15 which matches current bit width.
    return std::min<uint32_t>(best_k, 15u);
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
} // namespace

int Encoder::choose_rice_k(const std::vector<int32_t>& residual) {
    return static_cast<int>(estimate_initial_k(residual));
}

std::vector<uint8_t> Encoder::encode(const std::vector<int32_t>& pcm) {
    const int max_valid_order = (pcm.size() > 1)
        ? static_cast<int>(std::min<size_t>(32, pcm.size() - 1))
        : 0;

    struct CandidateEval {
        int target_order = 0;
        int used_order = 0;
        uint64_t bit_cost = std::numeric_limits<uint64_t>::max();
        long double energy = 0.0L;
        bool stable = false;
        std::vector<int16_t> coeffs_q15;
        std::vector<int32_t> residual;
    };

    struct DebugEval {
        int target_order = 0;
        int used_order = 0;
        uint64_t bit_cost = 0;
        bool stable = false;
        long double energy = 0.0L;
    };

    CandidateEval best;
    std::vector<DebugEval> debug_evals;
    debug_evals.reserve(sizeof(kOrderCandidates) / sizeof(kOrderCandidates[0]));

    for (int cand : kOrderCandidates) {
        if (cand > max_valid_order) continue;
        LPC lpc(cand);
        CandidateEval eval;
        eval.target_order = cand;

        int used_order = 0;
        long double energy = 0.0L;
        eval.stable = lpc.analyze_block_q15(pcm, eval.coeffs_q15, used_order, &energy);
        eval.used_order = used_order;
        eval.energy = energy;

        if (!eval.stable || eval.used_order == 0) {
            debug_evals.push_back({eval.target_order, eval.used_order, eval.bit_cost, eval.stable, eval.energy});
            continue;
        }

        eval.residual.resize(pcm.size());
        if (!pcm.empty()) {
            lpc.compute_residual_q15(pcm, eval.coeffs_q15, eval.residual);
        }

        eval.bit_cost = estimate_rice_bits(eval.residual);
        debug_evals.push_back({eval.target_order, eval.used_order, eval.bit_cost, eval.stable, eval.energy});

        if (eval.bit_cost < best.bit_cost ||
            (eval.bit_cost == best.bit_cost && eval.used_order < best.used_order)) {
            best = eval;
        }
    }

    if (best.used_order == 0) {
        const int fallback_order = std::max(1, std::min(this->order, max_valid_order));
        LPC lpc_fallback(fallback_order);
        int used_order = 0;
        long double energy = 0.0L;
        std::vector<int16_t> coeffs_q15;
        lpc_fallback.analyze_block_q15(pcm, coeffs_q15, used_order, &energy);
        std::vector<int32_t> residual(pcm.size());
        if (!pcm.empty()) {
            lpc_fallback.compute_residual_q15(pcm, coeffs_q15, residual);
        }
        best.target_order = fallback_order;
        best.used_order = used_order;
        best.energy = energy;
        best.coeffs_q15 = std::move(coeffs_q15);
        best.residual = std::move(residual);
        best.bit_cost = estimate_rice_bits(best.residual);
        best.stable = used_order > 0;
    }

    int chosen_order = best.used_order > 0 ? best.used_order : this->order;
    if (chosen_order > max_valid_order && max_valid_order > 0) {
        chosen_order = max_valid_order;
    }
    if (chosen_order <= 0) {
        chosen_order = 1;
    }

    std::vector<uint32_t> rice_unsigned(best.residual.size());
    std::vector<uint64_t> rice_prefix(best.residual.size());
    if (!best.residual.empty()) {
        SIMD::rice_unsigned_simd_or_scalar(best.residual.data(),
                                           rice_unsigned.data(),
                                           best.residual.size());
        SIMD::rice_prefix_sum_simd_or_scalar(rice_unsigned.data(),
                                             rice_prefix.data(),
                                             best.residual.size());
    }

    std::vector<uint32_t> rice_k_next(best.residual.size());
    for (size_t i = 0; i < best.residual.size(); ++i) {
        rice_k_next[i] = Rice::adapt_k(rice_prefix[i],
                                       static_cast<uint32_t>(i + 1));
    }

    int k = this->choose_rice_k(best.residual);
    uint32_t current_k = static_cast<uint32_t>(k);

    Rice rice;

    BitWriter bw;

    bw.write_bits(static_cast<uint32_t>(chosen_order), 8);
    bw.write_bits(k, 5);

    for (int i = 1; i <= chosen_order; ++i) {
        bw.write_bits(uint16_t(best.coeffs_q15[i]), 16);
    }

    for (size_t i = 0; i < best.residual.size(); ++i) {
        rice.encode(bw, best.residual[i], current_k);
        current_k = rice_k_next[i];
    }

    bw.flush_to_byte();

    if (this->debug_lpc) {
        std::cerr << "[debug-lpc] block=" << pcm.size()
                  << " energy=" << static_cast<double>(best.energy)
                  << " chosen_order=" << chosen_order
                  << " est_bits=" << best.bit_cost << "\n";
        for (const auto& e : debug_evals) {
            std::cerr << "  cand=" << e.target_order
                      << " used=" << e.used_order
                      << " bits=" << e.bit_cost
                      << (e.stable ? "" : " unstable")
                      << "\n";
        }
    }

    return bw.get_buffer();
}

} // namespace Block
