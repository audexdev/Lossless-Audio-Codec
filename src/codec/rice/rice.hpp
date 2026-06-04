#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/bitstream/bit_reader.hpp"

class Rice {
public:
    static constexpr uint32_t kDriftWindow = 256;
    static constexpr uint32_t kMicroWindow = 96;

    struct AdaptState {
        uint64_t previous_sum = 0;
        uint32_t window_index = 0;
        uint32_t micro_index = 0;
        uint32_t window_filled = 0;
        uint64_t window_sum = 0;
        uint16_t large_q_count = 0;
        uint16_t zero_q_count = 0;
        std::array<uint32_t, kDriftWindow> recent_u;
        std::array<uint8_t, kMicroWindow> large_flags;
        std::array<uint8_t, kMicroWindow> zero_flags;

        AdaptState() {
            recent_u.fill(0);
            large_flags.fill(0);
            zero_flags.fill(0);
        }
    };

    static void encode(BitWriter& w, int32_t value, uint32_t k);

    static bool decode(BitReader& r, uint32_t k, int32_t& value);

    // Compute optimal k from residual block
    static uint32_t compute_k(const std::vector<int32_t>& residuals);

    static uint32_t adapt_k(uint64_t sum, uint32_t count, AdaptState& state);

private:
    static uint32_t signed_to_unsigned(int32_t v);
    static int32_t unsigned_to_signed(uint32_t u);
};

inline uint32_t Rice::adapt_k(uint64_t sum, uint32_t count, AdaptState& state) {
    if (count == 0) return 0;

    constexpr uint32_t kMaxRiceK = 31;
    const uint64_t current_u = sum - state.previous_sum;
    state.previous_sum = sum;

    // Track micro-window flags first so we can evict the slot we will overwrite.
    const uint32_t micro_idx = state.micro_index;
    state.large_q_count -= state.large_flags[micro_idx];
    state.zero_q_count -= state.zero_flags[micro_idx];

    // Update drift window.
    if (state.window_filled < kDriftWindow) {
        ++state.window_filled;
    } else {
        state.window_sum -= state.recent_u[state.window_index];
    }
    state.recent_u[state.window_index] = static_cast<uint32_t>(current_u);
    state.window_sum += current_u;

    // Micro window flags.
    // Base mean and k estimate using integer rounding.
    const uint64_t mean = (sum + (count >> 1)) / count;
    const uint32_t k = (mean <= 1)
        ? 0u
        : std::min<uint32_t>(kMaxRiceK, std::bit_width(mean - 1u));

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
        const uint64_t local_mean = (state.window_filled == kDriftWindow)
            ? ((state.window_sum + (kDriftWindow >> 1)) >> 8)
            : ((state.window_sum + (state.window_filled >> 1)) / state.window_filled);
        // Increase k if local variation is significantly higher.
        if (local_mean * 3 > mean * 4) {
            bias = 1;
        } else if (local_mean * 4 + 3 < mean * 3) {
            bias = -1;
        }
    }

    // Micro adaptation on quotient distribution.
    if (state.window_index + 1 >= kMicroWindow || state.window_filled >= kMicroWindow) {
        const uint32_t window_size =
            (state.window_filled >= kMicroWindow) ? kMicroWindow : state.window_filled;
        if (state.large_q_count * 4 >= window_size * 3) {
            bias = std::min<int32_t>(bias + 1, 1);
        } else if (state.zero_q_count * 5 >= window_size * 4) {
            bias = std::max<int32_t>(bias - 1, -1);
        }
    }

    int32_t biased_k = static_cast<int32_t>(k) + bias;
    if (biased_k < 0) biased_k = 0;
    if (biased_k > 31) biased_k = 31;

    state.micro_index = (state.micro_index + 1u == kMicroWindow) ? 0u : state.micro_index + 1u;
    state.window_index = (state.window_index + 1u) & (kDriftWindow - 1u);
    return static_cast<uint32_t>(biased_k);
}
