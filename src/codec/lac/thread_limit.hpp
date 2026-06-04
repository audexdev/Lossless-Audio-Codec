#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace LAC {

inline size_t parse_thread_limit(const char* value) {
    if (value == nullptr || value[0] == '\0') return 0;
    for (const char* p = value; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            throw std::invalid_argument("LAC_THREADS must be a positive integer");
        }
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        throw std::invalid_argument("LAC_THREADS must be a positive integer");
    }
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument("LAC_THREADS is too large");
    }
    return static_cast<size_t>(parsed);
}

inline size_t resolve_thread_limit(size_t explicit_limit) {
    if (explicit_limit > 0) return explicit_limit;
    return parse_thread_limit(std::getenv("LAC_THREADS"));
}

} // namespace LAC
