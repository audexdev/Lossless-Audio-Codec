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

Encoder::Encoder(int order, bool debug_lpc, bool debug_zr)
    : order(order),
      debug_lpc(debug_lpc),
      debug_zr(debug_zr) {
    this->zero_run_enabled = true;
    this->partitioning_enabled = true;
}

void Encoder::set_zero_run_enabled(bool enabled) {
    this->zero_run_enabled = enabled;
}

void Encoder::set_debug_block_index(size_t index) {
    this->block_index = index;
}

void Encoder::set_partitioning_enabled(bool enabled) {
    this->partitioning_enabled = enabled;
}

void Encoder::set_debug_partitions(bool enabled) {
    this->debug_partitions = enabled;
}

namespace {
constexpr int kOrderCandidates[] = {4, 6, 8, 10, 12};
constexpr size_t kInitialScanCount = 256;
constexpr uint32_t kInitialMaxK = 12;
constexpr uint32_t kZeroRunMinRun = Block::ZERO_RUN_MIN_LENGTH;
constexpr uint32_t kZeroRunRunK = Block::ZERO_RUN_LENGTH_K;
constexpr uint32_t kBinTagZero = 0b00u;
constexpr uint32_t kBinTagOne = 0b01u;
constexpr uint32_t kBinTagTwo = 0b10u;
constexpr uint32_t kBinTagFallback = 0b11u;
constexpr uint8_t kPredictorFixed = 0;
constexpr uint8_t kPredictorFir = 1;
constexpr uint8_t kPredictorLpc = 2;
constexpr int kFirShift = 2;
constexpr int kFirTaps[] = {3, -1};

inline uint32_t unsigned_from_residual(int32_t r) {
    return static_cast<uint32_t>((r << 1) ^ (r >> 31));
}

inline uint64_t rice_bits_for_unsigned(uint32_t u, uint32_t k) {
    const uint32_t q = (k >= 31u) ? 0u : (u >> k);
    return static_cast<uint64_t>(q) + 1u + k;
}

inline uint32_t adapt_k_stateless(uint64_t sum, uint32_t count) {
    if (count == 0) return 0;
    const uint64_t mean = (sum + (count >> 1)) / count;
    uint32_t k = 0;
    while ((1u << k) < mean && k < 31u) {
        ++k;
    }
    return k;
}

inline void write_rice_unsigned(BitWriter& bw, uint32_t value, uint32_t k) {
    const uint32_t q = (k >= 31u) ? 0u : (value >> k);
    for (uint32_t i = 0; i < q; ++i) {
        bw.write_bit(1u);
    }
    bw.write_bit(0u);
    if (k > 0) {
        const uint32_t rem_mask = (1u << k) - 1u;
        bw.write_bits(value & rem_mask, static_cast<int>(k));
    }
}

inline uint32_t zigzag_encode(int32_t v) {
    return static_cast<uint32_t>((static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31));
}

inline uint8_t max_partition_order_for_block(uint32_t block_size) {
    uint8_t max_p = 0;
    for (uint8_t p = 1; p <= Block::MAX_PARTITION_ORDER; ++p) {
        const uint32_t base = block_size >> p;
        if (base < Block::MIN_PARTITION_SIZE) break;
        max_p = p;
    }
    return max_p;
}

inline std::vector<uint32_t> partition_sizes_for_block(uint32_t block_size, uint8_t partition_order) {
    std::vector<uint32_t> sizes;
    if (partition_order == 0) {
        sizes.push_back(block_size);
        return sizes;
    }
    const uint32_t base = block_size >> partition_order;
    if (base == 0) {
        sizes.push_back(block_size);
        return sizes;
    }
    const uint32_t num_parts = 1u << partition_order;
    sizes.assign(num_parts, base);
    const uint32_t used = base * (num_parts - 1u);
    sizes.back() = block_size - used;
    return sizes;
}

bool has_zero_run(const std::vector<int32_t>& residual) {
    size_t idx = 0;
    while (idx < residual.size()) {
        if (residual[idx] == 0) {
            size_t run = 0;
            while (idx + run < residual.size() && residual[idx + run] == 0) {
                ++run;
            }
            if (run >= kZeroRunMinRun) return true;
            idx += run;
        } else {
            ++idx;
        }
    }
    return false;
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
        bits += rice_bits_for_unsigned(u, k);
    }
    return bits;
}

uint64_t estimate_adaptive_rice_bits(const std::vector<int32_t>& residual,
                                     uint32_t initial_k,
                                     bool stateless = false) {
    uint64_t bits = 0;
    uint32_t current_k = initial_k;
    uint64_t sum_u = 0;
    for (size_t i = 0; i < residual.size(); ++i) {
        const uint32_t u = unsigned_from_residual(residual[i]);
        bits += rice_bits_for_unsigned(u, current_k);
        sum_u += u;
        current_k = stateless
            ? adapt_k_stateless(sum_u, static_cast<uint32_t>(i + 1))
            : Rice::adapt_k(sum_u, static_cast<uint32_t>(i + 1));
    }
    return bits;
}

uint64_t estimate_zerorun_bits(const std::vector<int32_t>& residual,
                               uint32_t initial_k,
                               bool stateless = false) {
    if (residual.empty()) return 0;
    uint64_t bits = 0;
    uint32_t current_k = initial_k;
    uint64_t sum_u = 0;
    uint32_t count = 0;

    size_t idx = 0;
    while (idx < residual.size()) {
        size_t run = 0;
        while (idx + run < residual.size() && residual[idx + run] == 0) {
            ++run;
        }
        if (run >= kZeroRunMinRun) {
            bits += 2; // tag
            bits += rice_bits_for_unsigned(static_cast<uint32_t>(run - kZeroRunMinRun), kZeroRunRunK);
            for (size_t j = 0; j < run; ++j) {
                ++count;
                current_k = stateless
                    ? adapt_k_stateless(sum_u, count)
                    : Rice::adapt_k(sum_u, count);
            }
            idx += run;
            continue;
        }

        const uint32_t u = unsigned_from_residual(residual[idx]);
        const uint32_t escape_threshold = 1u << std::min<uint32_t>(24, current_k + 3u);
        if (u > escape_threshold) {
            bits += 2; // tag
            bits += 32; // escape payload
            sum_u += u;
            ++count;
            current_k = stateless
                ? adapt_k_stateless(sum_u, count)
                : Rice::adapt_k(sum_u, count);
            ++idx;
            continue;
        }

        // Normal token: single residual only.
        bits += 2; // tag
        bits += rice_bits_for_unsigned(u, current_k);
        sum_u += u;
        ++count;
        current_k = stateless
            ? adapt_k_stateless(sum_u, count)
            : Rice::adapt_k(sum_u, count);
        ++idx;
    }
    return bits;
}

uint64_t estimate_binning_bits(const std::vector<int32_t>& residual,
                               uint32_t initial_k,
                               bool stateless = false) {
    if (residual.empty()) return 0;
    uint64_t bits = 0;
    uint32_t current_k = initial_k;
    uint64_t sum_u = 0;
    uint32_t count = 0;

    for (int32_t v : residual) {
        const uint32_t u = unsigned_from_residual(v);
        if (v == 0) {
            bits += 2; // 00
        } else if (v == 1 || v == -1) {
            bits += 3; // 01 + sign
        } else if (v == 2 || v == -2) {
            bits += 3; // 10 + sign
        } else {
            bits += 2; // fallback tag
            bits += rice_bits_for_unsigned(u, current_k);
        }
        sum_u += u;
        ++count;
        current_k = stateless
            ? adapt_k_stateless(sum_u, count)
            : Rice::adapt_k(sum_u, count);
    }
    return bits;
}

void compute_fixed_residual(const std::vector<int32_t>& pcm, int order, std::vector<int32_t>& residual) {
    residual.resize(pcm.size(), 0);
    if (order == 0) {
        residual = pcm;
        return;
    }
    for (int i = 0; i < order && i < static_cast<int>(pcm.size()); ++i) {
        residual[i] = pcm[i];
    }
    for (size_t i = static_cast<size_t>(order); i < pcm.size(); ++i) {
        int64_t pred = 0;
        switch (order) {
            case 1:
                pred = pcm[i - 1];
                break;
            case 2:
                pred = 2LL * pcm[i - 1] - pcm[i - 2];
                break;
            case 3:
                pred = 3LL * pcm[i - 1] - 3LL * pcm[i - 2] + pcm[i - 3];
                break;
            case 4:
                pred = 4LL * pcm[i - 1] - 6LL * pcm[i - 2] + 4LL * pcm[i - 3] - pcm[i - 4];
                break;
            default:
                pred = 0;
                break;
        }
        residual[i] = static_cast<int32_t>(pcm[i] - pred);
    }
}

void compute_fir_residual(const std::vector<int32_t>& pcm, int taps, std::vector<int32_t>& residual) {
    residual.resize(pcm.size(), 0);
    for (int i = 0; i < taps && i < static_cast<int>(pcm.size()); ++i) {
        residual[i] = pcm[i];
    }
    for (size_t i = static_cast<size_t>(taps); i < pcm.size(); ++i) {
        int64_t pred = 0;
        pred += static_cast<int64_t>(kFirTaps[0]) * static_cast<int64_t>(pcm[i - 1]);
        pred += static_cast<int64_t>(kFirTaps[1]) * static_cast<int64_t>(pcm[i - 2]);
        pred >>= kFirShift;
        residual[i] = static_cast<int32_t>(static_cast<int64_t>(pcm[i]) - pred);
    }
}

} // namespace

int Encoder::choose_rice_k(const std::vector<int32_t>& residual) {
    return static_cast<int>(estimate_initial_k(residual));
}

bool Encoder::estimate_bits(const std::vector<int32_t>& pcm,
                            uint64_t& bits_normal,
                            uint64_t& bits_zr,
                            uint64_t& bits_bin) {
    bits_normal = 0;
    bits_zr = 0;
    bits_bin = 0;
    const int max_valid_order = (pcm.size() > 1)
        ? static_cast<int>(std::min<size_t>(32, pcm.size() - 1))
        : 0;

    struct CandidateEval {
        int target_order = 0;
        int used_order = 0;
        uint64_t bit_cost = std::numeric_limits<uint64_t>::max();
        uint64_t zr_bits = std::numeric_limits<uint64_t>::max();
        uint64_t bin_bits = std::numeric_limits<uint64_t>::max();
        long double energy = 0.0L;
        bool stable = false;
        std::vector<int16_t> coeffs_q15;
        std::vector<int32_t> residual;
    };

    CandidateEval best;
    uint64_t best_metric = std::numeric_limits<uint64_t>::max();

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
            continue;
        }

        eval.residual.resize(pcm.size());
        if (!pcm.empty()) {
            lpc.compute_residual_q15(pcm, eval.coeffs_q15, eval.residual);
        }

        uint32_t adaptive_k = estimate_initial_k(eval.residual);
        eval.bit_cost = estimate_adaptive_rice_bits(eval.residual, adaptive_k);
        const bool allow_zr = this->zero_run_enabled && has_zero_run(eval.residual);
        eval.zr_bits = allow_zr
            ? estimate_zerorun_bits(eval.residual, adaptive_k)
            : eval.bit_cost;
        eval.bin_bits = estimate_binning_bits(eval.residual, adaptive_k);
        uint64_t candidate_score = eval.bit_cost;
        if (this->zero_run_enabled && has_zero_run(eval.residual)) {
            candidate_score = std::min(candidate_score, eval.zr_bits);
        }
        candidate_score = std::min(candidate_score, eval.bin_bits);

        if (candidate_score < best_metric ||
            (candidate_score == best_metric && eval.used_order < best.used_order)) {
            best = eval;
            best_metric = candidate_score;
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
        uint32_t adaptive_k = estimate_initial_k(best.residual);
        best.bit_cost = estimate_adaptive_rice_bits(best.residual, adaptive_k);
        best.stable = used_order > 0;
        const bool allow_zr = this->zero_run_enabled && has_zero_run(best.residual);
        best.zr_bits = allow_zr
            ? estimate_zerorun_bits(best.residual, adaptive_k)
            : best.bit_cost;
        best_metric = (allow_zr ? std::min(best.bit_cost, best.zr_bits) : best.bit_cost);
        best.bin_bits = estimate_binning_bits(best.residual, adaptive_k);
        best_metric = std::min(best_metric, best.bin_bits);
    }

    uint32_t best_k = estimate_initial_k(best.residual);
    bits_normal = estimate_adaptive_rice_bits(best.residual, best_k);
    const bool allow_zr = this->zero_run_enabled && has_zero_run(best.residual);
    bits_zr = allow_zr
        ? estimate_zerorun_bits(best.residual, best_k)
        : bits_normal;
    bits_bin = estimate_binning_bits(best.residual, best_k);
    return !best.residual.empty();
}

std::vector<uint8_t> Encoder::encode(const std::vector<int32_t>& pcm) {
    const int max_valid_order = (pcm.size() > 1)
        ? static_cast<int>(std::min<size_t>(32, pcm.size() - 1))
        : 0;

    struct PredictorEval {
        uint8_t predictor_type = kPredictorLpc;
        int order_param = 0; // fixed order, fir taps, or lpc order
        int used_order = 0; // for LPC stability info
        uint64_t rice_bits = std::numeric_limits<uint64_t>::max();
        uint64_t zr_bits = std::numeric_limits<uint64_t>::max();
        uint64_t bin_bits = std::numeric_limits<uint64_t>::max();
        uint64_t best_bits = std::numeric_limits<uint64_t>::max();
        long double energy = 0.0L;
        bool stable = true;
        std::vector<int16_t> coeffs_q15;
        std::vector<int32_t> residual;
    };

    std::vector<PredictorEval> candidates;

    // Fixed predictors 0-4.
    for (int fo = 0; fo <= 4; ++fo) {
        PredictorEval ev;
        ev.predictor_type = kPredictorFixed;
        ev.order_param = fo;
        compute_fixed_residual(pcm, fo, ev.residual);
        uint32_t k0 = estimate_initial_k(ev.residual);
        ev.rice_bits = estimate_adaptive_rice_bits(ev.residual, k0);
        ev.zr_bits = estimate_zerorun_bits(ev.residual, k0);
        ev.bin_bits = estimate_binning_bits(ev.residual, k0);
        ev.best_bits = std::min(ev.rice_bits, std::min(ev.zr_bits, ev.bin_bits));
        candidates.push_back(std::move(ev));
    }

    // FIR predictor (2 taps predefined).
    {
        PredictorEval ev;
        ev.predictor_type = kPredictorFir;
        ev.order_param = 2;
        compute_fir_residual(pcm, ev.order_param, ev.residual);
        uint32_t k0 = estimate_initial_k(ev.residual);
        ev.rice_bits = estimate_adaptive_rice_bits(ev.residual, k0);
        ev.zr_bits = estimate_zerorun_bits(ev.residual, k0);
        ev.bin_bits = estimate_binning_bits(ev.residual, k0);
        ev.best_bits = std::min(ev.rice_bits, std::min(ev.zr_bits, ev.bin_bits));
        candidates.push_back(std::move(ev));
    }

    // LPC predictors.
    for (int cand : kOrderCandidates) {
        if (cand > max_valid_order) continue;
        LPC lpc(cand);
        PredictorEval ev;
        ev.predictor_type = kPredictorLpc;
        ev.order_param = cand;

        int used_order = 0;
        long double energy = 0.0L;
        ev.stable = lpc.analyze_block_q15(pcm, ev.coeffs_q15, used_order, &energy);
        ev.used_order = used_order;
        ev.energy = energy;
        if (!ev.stable || ev.used_order == 0) {
            continue;
        }

        ev.residual.resize(pcm.size());
        if (!pcm.empty()) {
            lpc.compute_residual_q15(pcm, ev.coeffs_q15, ev.residual);
        }
        uint32_t k0 = estimate_initial_k(ev.residual);
        ev.rice_bits = estimate_adaptive_rice_bits(ev.residual, k0);
        ev.zr_bits = estimate_zerorun_bits(ev.residual, k0);
        ev.bin_bits = estimate_binning_bits(ev.residual, k0);
        ev.best_bits = std::min(ev.rice_bits, std::min(ev.zr_bits, ev.bin_bits));
        candidates.push_back(std::move(ev));
    }

    // Ensure at least one candidate (fallback fixed-0).
    if (candidates.empty()) {
        PredictorEval ev;
        ev.predictor_type = kPredictorFixed;
        ev.order_param = 0;
        ev.residual = pcm;
        uint32_t k0 = estimate_initial_k(ev.residual);
        ev.rice_bits = estimate_adaptive_rice_bits(ev.residual, k0);
        ev.zr_bits = estimate_zerorun_bits(ev.residual, k0);
        ev.bin_bits = estimate_binning_bits(ev.residual, k0);
        ev.best_bits = std::min(ev.rice_bits, std::min(ev.zr_bits, ev.bin_bits));
        candidates.push_back(std::move(ev));
    }

    PredictorEval best = candidates.front();
    for (const auto& ev : candidates) {
        if (ev.best_bits < best.best_bits ||
            (ev.best_bits == best.best_bits && ev.predictor_type < best.predictor_type)) {
            best = ev;
        }
    }

    const int chosen_order = (best.predictor_type == kPredictorLpc)
        ? std::max(1, std::min(best.order_param, max_valid_order))
        : best.order_param;

    struct PartitionChoice {
        uint8_t residual_mode = 0; // 0 = rice, 1 = zero-run, 2 = bin
        uint32_t initial_k = 0;
        uint64_t bits = 0;
        uint32_t length = 0;
    };

    const uint32_t block_size = static_cast<uint32_t>(best.residual.size());
    const uint32_t base_initial_k = static_cast<uint32_t>(this->choose_rice_k(best.residual));
    uint64_t base_bits_normal = estimate_adaptive_rice_bits(best.residual, base_initial_k);
    const bool has_run = has_zero_run(best.residual);
    const bool allow_zr_global = this->zero_run_enabled && has_run;
    uint64_t base_bits_zr = allow_zr_global ? estimate_zerorun_bits(best.residual, base_initial_k) : base_bits_normal;
    uint64_t base_bits_bin = estimate_binning_bits(best.residual, base_initial_k);
    uint8_t base_mode = 0; // 0=rice,1=zr,2=bin
    uint64_t base_bits_best = base_bits_normal;
    if (allow_zr_global && base_bits_zr <= base_bits_best) {
        base_bits_best = base_bits_zr;
        base_mode = 1;
    }
    if (base_bits_bin < base_bits_best) {
        base_bits_best = base_bits_bin;
        base_mode = 2;
    }
    if (this->debug_zr && this->zero_run_enabled) {
        std::cerr << "[zr-est] block=" << this->block_index
                  << " normal=" << base_bits_normal
                  << " zr=" << base_bits_zr
                  << " bin=" << base_bits_bin
                  << " chosen=" << static_cast<int>(base_mode)
                  << " has_run=" << has_run
                  << "\n";
    }

    PartitionChoice legacy_choice{
        base_mode,
        base_initial_k,
        base_bits_best,
        block_size
    };

    auto metadata_bits = [&](uint8_t partition_order, uint32_t partition_count) -> uint64_t {
        uint64_t bits = 8; // control byte always present
        bits += static_cast<uint64_t>(partition_count) * 7ull; // 2-bit mode + 5-bit k per partition
        return bits;
    };

    uint8_t best_partition_order = 0;
    std::vector<PartitionChoice> best_partitions = {legacy_choice};
    uint64_t best_total_bits = legacy_choice.bits + metadata_bits(0, 1);
    best_total_bits += (8u - (best_total_bits & 7u)) & 7u;

    if (this->partitioning_enabled && block_size >= Block::MIN_PARTITION_SIZE) {
        const uint8_t max_p = max_partition_order_for_block(block_size);
        const bool allow_partition_zr = allow_zr_global;
        for (uint8_t p = 0; p <= max_p; ++p) {
            const auto sizes = partition_sizes_for_block(block_size, p);
            if (sizes.empty()) continue;
            std::vector<PartitionChoice> choices;
            choices.reserve(sizes.size());
            uint64_t bits_sum = 0;
            size_t offset = 0;
            for (uint32_t len : sizes) {
                PartitionChoice pc;
                pc.length = len;
                const std::vector<int32_t> segment(best.residual.begin() + offset, best.residual.begin() + offset + len);
                pc.initial_k = estimate_initial_k(segment);
                const uint64_t normal_bits = estimate_adaptive_rice_bits(segment, pc.initial_k, true);
                const uint64_t bin_bits = estimate_binning_bits(segment, pc.initial_k, true);
                const bool allow_zr = allow_partition_zr && has_zero_run(segment);
                const uint64_t zr_bits = allow_zr ? estimate_zerorun_bits(segment, pc.initial_k, true) : normal_bits;
                pc.residual_mode = 0;
                pc.bits = normal_bits;
                if (allow_zr && zr_bits < pc.bits) {
                    pc.residual_mode = 1;
                    pc.bits = zr_bits;
                }
                if (bin_bits < pc.bits) {
                    pc.residual_mode = 2;
                    pc.bits = bin_bits;
                }
                bits_sum += pc.bits;
                choices.push_back(pc);
                offset += len;
            }
            uint64_t meta = metadata_bits(p, static_cast<uint32_t>(choices.size()));
            uint64_t total = bits_sum + meta;
            total += (8u - (total & 7u)) & 7u;
            if (this->debug_partitions) {
                std::cerr << "[part-est] block=" << this->block_index
                          << " p=" << static_cast<uint32_t>(p)
                          << " bits=" << total
                          << " partitions=" << choices.size()
                          << "\n";
            }
            if (total < best_total_bits ||
                (total == best_total_bits && p < best_partition_order)) {
                best_total_bits = total;
                best_partitions = choices;
                best_partition_order = p;
            }
        }
        if (this->debug_partitions) {
            std::cerr << "[part-choose] block=" << this->block_index
                      << " best_p=" << static_cast<uint32_t>(best_partition_order)
                      << " bits=" << best_total_bits
                      << "\n";
        }
    }

    Rice rice;
    BitWriter bw;
    const bool use_stateless_adapt = (best_partition_order > 0);

    if (this->debug_partitions && best_partition_order > 0) {
        std::cerr << "[part-plan] block=" << this->block_index
                  << " order=" << static_cast<uint32_t>(best_partition_order)
                  << " parts=" << best_partitions.size();
        size_t dbg_offset = 0;
        for (size_t i = 0; i < best_partitions.size(); ++i) {
            const auto& p = best_partitions[i];
            std::cerr << " [" << i << " mode=" << static_cast<int>(p.residual_mode)
                      << " k=" << p.initial_k
                      << " len=" << p.length
                      << " bits=" << p.bits << "]";
            size_t zero_count = 0;
            int32_t max_abs = 0;
            for (size_t j = 0; j < p.length && dbg_offset + j < best.residual.size(); ++j) {
                int32_t v = best.residual[dbg_offset + j];
                if (v == 0) ++zero_count;
                int32_t abs_v = (v >= 0) ? v : -v;
                if (abs_v > max_abs) max_abs = abs_v;
            }
            std::cerr << " zc=" << zero_count << " max=" << max_abs;
            dbg_offset += p.length;
        }
        std::cerr << "\n";
    }

    auto encode_rice_partition = [&](size_t start, size_t length, uint32_t initial_k_value) {
        uint32_t current_k = initial_k_value;
        uint64_t sum_u = 0;
        for (size_t i = 0; i < length; ++i) {
            const size_t pos = start + i;
            const int32_t v = best.residual[pos];
            const uint32_t u = unsigned_from_residual(v);
            rice.encode(bw, v, current_k);
            sum_u += u;
            current_k = use_stateless_adapt
                ? adapt_k_stateless(sum_u, static_cast<uint32_t>(i + 1))
                : Rice::adapt_k(sum_u, static_cast<uint32_t>(i + 1));
        }
    };

    auto encode_bin_partition = [&](size_t start, size_t length, uint32_t initial_k_value, size_t part_index) {
        uint32_t current_k = initial_k_value;
        uint64_t sum_u = 0;
        uint32_t count = 0;
        size_t token_idx = 0;

        for (size_t i = 0; i < length; ++i) {
            const size_t pos = start + i;
            const int32_t v = best.residual[pos];
            const uint32_t u = unsigned_from_residual(v);
            if (v == 0) {
                bw.write_bits(kBinTagZero, 2);
                if (this->debug_partitions && token_idx < 12) {
                    std::cerr << "[part-enc] p=" << part_index
                              << " tok=" << token_idx
                              << " tag=bin0"
                              << " k=" << current_k << "\n";
                }
            } else if (v == 1 || v == -1) {
                bw.write_bits(kBinTagOne, 2);
                bw.write_bit(v < 0 ? 1u : 0u);
                if (this->debug_partitions && token_idx < 12) {
                    std::cerr << "[part-enc] p=" << part_index
                              << " tok=" << token_idx
                              << " tag=bin1"
                              << " sign=" << (v < 0 ? "-" : "+")
                              << " k=" << current_k << "\n";
                }
            } else if (v == 2 || v == -2) {
                bw.write_bits(kBinTagTwo, 2);
                bw.write_bit(v < 0 ? 1u : 0u);
                if (this->debug_partitions && token_idx < 12) {
                    std::cerr << "[part-enc] p=" << part_index
                              << " tok=" << token_idx
                              << " tag=bin2"
                              << " sign=" << (v < 0 ? "-" : "+")
                              << " k=" << current_k << "\n";
                }
            } else {
                bw.write_bits(kBinTagFallback, 2);
                rice.encode(bw, v, current_k);
                if (this->debug_partitions && token_idx < 12) {
                    std::cerr << "[part-enc] p=" << part_index
                              << " tok=" << token_idx
                              << " tag=bin-fb"
                              << " k=" << current_k
                              << " u=" << u << "\n";
                }
            }
            sum_u += u;
            ++count;
            current_k = use_stateless_adapt
                ? adapt_k_stateless(sum_u, count)
                : Rice::adapt_k(sum_u, count);
            ++token_idx;
        }
    };

    auto encode_zr_partition = [&](size_t start, size_t length, uint32_t initial_k_value, size_t part_index) {
        constexpr uint32_t kTagNormal = 0b00u;
        constexpr uint32_t kTagRun = 0b01u;
        constexpr uint32_t kTagEscape = 0b10u;

        uint32_t current_k = initial_k_value;
        uint64_t sum_u = 0;
        uint32_t count = 0;
        size_t token_idx = 0;

        size_t idx = start;
        const size_t end = start + length;
        while (idx < end) {
            size_t run = 0;
            while (idx + run < end && best.residual[idx + run] == 0) {
                ++run;
            }
            if (run >= kZeroRunMinRun) {
            if (this->debug_zr) {
                std::cerr << "[zr-enc-token] block=" << this->block_index
                          << " idx=" << idx
                          << " tag=run"
                          << " val=" << run
                          << "\n";
            }
                if (this->debug_partitions && token_idx < 12) {
                    std::cerr << "[part-enc] p=" << part_index
                              << " tok=" << token_idx
                              << " tag=run len=" << run
                              << " k=" << current_k << "\n";
                }
                bw.write_bits(kTagRun, 2);
                write_rice_unsigned(bw, static_cast<uint32_t>(run - kZeroRunMinRun), kZeroRunRunK);
                for (size_t j = 0; j < run; ++j) {
                    ++count;
                    current_k = use_stateless_adapt
                        ? adapt_k_stateless(sum_u, count)
                        : Rice::adapt_k(sum_u, count);
                }
                idx += run;
                ++token_idx;
                continue;
            }

            const uint32_t u = unsigned_from_residual(best.residual[idx]);
            const uint32_t escape_threshold = 1u << std::min<uint32_t>(24, current_k + 3u);
            if (u > escape_threshold) {
            if (this->debug_zr) {
                std::cerr << "[zr-enc-token] block=" << this->block_index
                          << " idx=" << idx
                          << " tag=escape"
                          << " val=" << best.residual[idx]
                          << "\n";
            }
                if (this->debug_partitions && token_idx < 12) {
                    std::cerr << "[part-enc] p=" << part_index
                              << " tok=" << token_idx
                              << " tag=esc"
                              << " k=" << current_k
                              << " u=" << u << "\n";
                }
                bw.write_bits(kTagEscape, 2);
                bw.write_bits(zigzag_encode(best.residual[idx]), 32);
                sum_u += u;
                ++count;
                current_k = use_stateless_adapt
                    ? adapt_k_stateless(sum_u, count)
                    : Rice::adapt_k(sum_u, count);
                ++idx;
                ++token_idx;
                continue;
            }

            // Normal token encodes exactly one residual.
            bw.write_bits(kTagNormal, 2);
            if (this->debug_partitions && token_idx < 12) {
                std::cerr << "[part-enc] p=" << part_index
                          << " tok=" << token_idx
                          << " tag=norm n=1"
                          << " k=" << current_k << "\n";
            }
            if (this->debug_zr) {
                std::cerr << "[zr-enc-token] block=" << this->block_index
                          << " idx=" << idx
                          << " tag=normal"
                          << " val=" << best.residual[idx]
                          << "\n";
            }
            rice.encode(bw, best.residual[idx], current_k);
            sum_u += u;
            ++count;
            current_k = use_stateless_adapt
                ? adapt_k_stateless(sum_u, count)
                : Rice::adapt_k(sum_u, count);
            ++idx;
            ++token_idx;
        }
    };

    uint8_t control_mode = static_cast<uint8_t>(best_partitions.empty() ? 0 : best_partitions.front().residual_mode);
    uint8_t control_byte = static_cast<uint8_t>((control_mode & 0x03u) << 5);
    if (best_partition_order > 0) {
        control_byte |= Block::PARTITION_FLAG;
        control_byte |= static_cast<uint8_t>((best_partition_order & Block::PARTITION_ORDER_MASK) << Block::PARTITION_ORDER_SHIFT);
    }

    // Per-channel layout:
    // predictor_type (8) | predictor_order (8) | LPC coeffs | residual_control (8) |
    // [residual_mode:2 | initial_k:5] * partitions | residual bitstream | byte align.
    bw.write_bits(static_cast<uint32_t>(best.predictor_type), 8);
    bw.write_bits(static_cast<uint32_t>(chosen_order), 8);
    if (best.predictor_type == kPredictorLpc) {
        for (int i = 1; i <= chosen_order; ++i) {
            bw.write_bits(uint16_t(best.coeffs_q15[i]), 16);
        }
    }

    bw.write_bits(control_byte, 8);
    for (const auto& part : best_partitions) {
        bw.write_bits(part.residual_mode, 2);
        bw.write_bits(part.initial_k, 5);
    }

    size_t offset = 0;
    size_t part_idx = 0;
    for (const auto& part : best_partitions) {
        if (this->debug_partitions && best_partition_order > 0) {
            std::cerr << "[part-samples] idx=" << part_idx << " first=";
            for (size_t j = 0; j < std::min<size_t>(8, part.length) && offset + j < best.residual.size(); ++j) {
                std::cerr << best.residual[offset + j] << (j + 1 < std::min<size_t>(8, part.length) ? "," : "");
            }
            std::cerr << "\n";
        }
        if (part.residual_mode == 0) {
            encode_rice_partition(offset, part.length, part.initial_k);
        } else if (part.residual_mode == 1) {
            encode_zr_partition(offset, part.length, part.initial_k, part_idx);
        } else {
            encode_bin_partition(offset, part.length, part.initial_k, part_idx);
        }
        offset += part.length;
        ++part_idx;
    }

    bw.flush_to_byte();

    if (this->debug_lpc) {
        std::cerr << "[debug-lpc] block=" << pcm.size()
                  << " energy=" << static_cast<double>(best.energy)
                  << " chosen_order=" << chosen_order
                  << " predictor=" << static_cast<int>(best.predictor_type)
                  << " est_bits=" << best.best_bits
                  << " rice_bits=" << best.rice_bits
                  << " zr_bits=" << best.zr_bits
                  << " bin_bits=" << best.bin_bits
                  << " part_order=" << static_cast<uint32_t>(best_partition_order)
                  << "\n";
    }

    return bw.get_buffer();
}

} // namespace Block
