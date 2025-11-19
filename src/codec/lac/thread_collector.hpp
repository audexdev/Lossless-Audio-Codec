#pragma once
#include <set>
#include <mutex>
#include <thread>

namespace LAC {

class ThreadCollector {
public:
    void record(std::thread::id id) {
        std::lock_guard<std::mutex> lock(mutex_);
        ids_.insert(id);
    }

    std::set<std::thread::id> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ids_;
    }

private:
    mutable std::mutex mutex_;
    std::set<std::thread::id> ids_;
};

} // namespace LAC

