#include "rice.hpp"
#include <algorithm>
#include <array>

uint32_t Rice::signed_to_unsigned(int32_t v) {
    return (uint32_t)((v << 1) ^ (v >> 31));
}

int32_t Rice::unsigned_to_signed(uint32_t u) {
    return (u >> 1) ^ -((int32_t)(u & 1));
}

void Rice::encode(BitWriter& w, int32_t value, uint32_t k) {
    uint32_t u = signed_to_unsigned(value);

    uint32_t q = u >> k;
    uint32_t r = u & ((1u << k) - 1);

    for (uint32_t i = 0; i < q; ++i) {
        w.write_bit(1);
    }
    w.write_bit(0);

    if (k > 0) {
        w.write_bits(r, k);
    }
}

int32_t Rice::decode(BitReader& r, uint32_t k) {
    uint32_t q = 0;
    while (r.read_bit() == 1) {
        q++;
    }

    uint32_t u = q << k;

    uint32_t rem = 0;
    if (k > 0) {
        rem = r.read_bits(k);
    }

    u |= rem;

    return unsigned_to_signed(u);
}

uint32_t Rice::compute_k(const std::vector<int32_t>& residuals) {
    uint64_t sum = 0;
    for (int32_t v : residuals) {
        uint32_t u = signed_to_unsigned(v);
        sum += u;
    }

    if (residuals.empty()) return 0;

    double mean = double(sum) / double(residuals.size());

    uint32_t k = 0;
    while ((1u << k) < (uint32_t)(mean + 0.5) && k < 31) {
        k++;
    }

    return k;
}

namespace {
constexpr uint32_t kMaxRiceK = 31;
constexpr uint32_t kDriftWindow = 256;
constexpr uint32_t kMicroWindow = 96;

struct AdaptState {
    uint64_t previous_sum;
    uint32_t window_index;
    uint32_t window_filled;
    uint64_t window_sum;
    uint16_t large_q_count;
    uint16_t zero_q_count;
    std::array<uint32_t, kDriftWindow> recent_u;
    std::array<uint8_t, kMicroWindow> large_flags;
    std::array<uint8_t, kMicroWindow> zero_flags;

    void reset() {
        previous_sum = 0;
        window_index = 0;
        window_filled = 0;
        window_sum = 0;
        large_q_count = 0;
        zero_q_count = 0;
    }
};
} // namespace

uint32_t Rice::adapt_k(uint64_t sum, uint32_t count) {
    if (count == 0) return 0;

    // Thread-local state so different blocks encoded/decoded in parallel do not collide.
    thread_local AdaptState state;
    if (count == 1) {
        state.reset();
        state.recent_u.fill(0);
        state.large_flags.fill(0);
        state.zero_flags.fill(0);
    }

    const uint64_t current_u = sum - state.previous_sum;
    state.previous_sum = sum;

    // Track micro-window flags first so we can evict the slot we will overwrite.
    const uint32_t micro_idx = (count - 1) % kMicroWindow;
    state.large_q_count -= state.large_flags[micro_idx];
    state.zero_q_count -= state.zero_flags[micro_idx];

    // Update drift window.
    if (state.window_filled < kDriftWindow) {
        state.window_filled++;
    } else {
        state.window_sum -= state.recent_u[state.window_index];
    }
    state.recent_u[state.window_index] = static_cast<uint32_t>(current_u);
    state.window_sum += current_u;

    // Micro window flags.
    // Base mean and k estimate using integer rounding.
    const uint64_t mean = (sum + (count >> 1)) / count;
    uint32_t k = 0;
    while ((1u << k) < mean && k < kMaxRiceK) {
        ++k;
    }

    const uint32_t q_base = static_cast<uint32_t>((k >= kMaxRiceK) ? 0u : (current_u >> k));
    const uint8_t is_large = static_cast<uint8_t>(q_base > 3u);
    const uint8_t is_zero = static_cast<uint8_t>(q_base == 0u);

    state.large_q_count += is_large;
    state.zero_q_count += is_zero;
    state.large_flags[micro_idx] = is_large;
    state.zero_flags[micro_idx] = is_zero;

    // Drift correction: compare local window mean against global mean.
    int32_t bias = 0;
    if (state.window_filled > 0 && mean > 0) {
        const uint64_t local_mean = (state.window_sum + (state.window_filled >> 1)) / state.window_filled;
        // Increase k if local variation is significantly higher.
        if (local_mean * 3 > mean * 4) { // local > 1.33 * global
            bias = 1;
        } else if (local_mean * 4 + 3 < mean * 3) { // local < 0.75 * global
            bias = -1;
        }
    }

    // Micro adaptation on quotient distribution.
    if (state.window_index + 1 >= kMicroWindow || state.window_filled >= kMicroWindow) {
        const uint32_t window_size = std::min<uint32_t>(state.window_filled, kMicroWindow);
        if (state.large_q_count * 4 >= window_size * 3) { // many large quotients
            bias = std::min<int32_t>(bias + 1, 1);
        } else if (state.zero_q_count * 5 >= window_size * 4) { // mostly zeros
            bias = std::max<int32_t>(bias - 1, -1);
        }
    }

    int32_t biased_k = static_cast<int32_t>(k) + bias;
    if (biased_k < 0) biased_k = 0;
    if (biased_k > 31) biased_k = 31;

    state.window_index = (state.window_index + 1) % kDriftWindow;
    return static_cast<uint32_t>(biased_k);
}
