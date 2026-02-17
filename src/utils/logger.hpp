#pragma once
#include <cstdlib>
#include <iostream>

namespace Logger {
inline bool trace_enabled() {
#ifndef NDEBUG
    static int enabled = []() {
        const char* val = std::getenv("LAC_TRACE");
        return (val && val[0] == '1') ? 1 : 0;
    }();
    return enabled != 0;
#else
    return false;
#endif
}

inline bool debug_zr_enabled() {
#ifndef NDEBUG
    static int enabled = []() {
        const char* val = std::getenv("LAC_DEBUG_ZR");
        return (val != nullptr) ? 1 : 0;
    }();
    return enabled != 0;
#else
    return false;
#endif
}

inline bool debug_part_enabled() {
#ifndef NDEBUG
    static int enabled = []() {
        const char* val = std::getenv("LAC_DEBUG_PART");
        return (val != nullptr) ? 1 : 0;
    }();
    return enabled != 0;
#else
    return false;
#endif
}
} // namespace Logger

#ifdef NDEBUG
#define LAC_DEBUG_LOG(expr) do {} while (0)
#define LAC_TRACE_LOG(expr) do {} while (0)
#define LAC_DEBUG_ZR_ENABLED() false
#define LAC_DEBUG_PART_ENABLED() false
#else
#define LAC_DEBUG_LOG(expr) do { std::cerr << expr; } while (0)
#define LAC_TRACE_LOG(expr) do { if (::Logger::trace_enabled()) { std::cerr << expr; } } while (0)
#define LAC_DEBUG_ZR_ENABLED() (::Logger::debug_zr_enabled())
#define LAC_DEBUG_PART_ENABLED() (::Logger::debug_part_enabled())
#endif
