#include "decoder.hpp"
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>

namespace LAC {

namespace {
constexpr uint32_t MAX_BLOCK_COUNT = 10'000'000;
constexpr uint64_t MAX_TOTAL_SAMPLES = 6'912'000'000ULL; // 10 hours @ 192 kHz

struct BlockDecodeResult {
    bool ok = true;
    size_t index = 0;
    std::vector<int32_t> left;
    std::vector<int32_t> right;
};
}

Decoder::Decoder(ThreadCollector* collector) : collector(collector) {}

bool Decoder::decode(const uint8_t* data, size_t size, std::vector<int32_t>& left, std::vector<int32_t>& right, FrameHeader* out_header) {
    left.clear();
    right.clear();

    FrameHeader hdr;
    size_t header_bytes = 0;
    if (!FrameHeader::parse(data, size, hdr, header_bytes))
        return false;

    BitReader br(data + header_bytes, size - header_bytes);

    if (br.bits_remaining() < 32) return false;
    uint32_t block_count = br.read_bits(32);
    if (br.has_error() || block_count == 0 || block_count > MAX_BLOCK_COUNT) return false;
    if (static_cast<uint64_t>(block_count) * 32ULL > br.bits_remaining()) return false;

    std::vector<uint32_t> block_sizes(block_count);
    uint64_t total_samples = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        if (br.bits_remaining() < 32) return false;
        uint32_t sz = br.read_bits(32);
        if (br.has_error() || sz == 0 || sz > Block::MAX_BLOCK_SIZE) return false;
        total_samples += sz;
        if (total_samples > MAX_TOTAL_SAMPLES) return false;
        block_sizes[i] = sz;
    }

    const bool isStereo = (hdr.channels == 2);
    const bool useMidSide = isStereo && (hdr.stereo_mode == 1);

    std::vector<size_t> blockOffsets(block_count);
    size_t running = 0;
    for (uint32_t i = 0; i < block_count; ++i) {
        blockOffsets[i] = running;
        running += block_sizes[i];
    }

    left.assign(running, 0);
    if (isStereo) right.assign(running, 0);
    else right.clear();

    std::mutex br_mutex;
    std::condition_variable br_cv;
    size_t next_block = 0;

    auto launch_policy = std::thread::hardware_concurrency() > 1
        ? std::launch::async
        : std::launch::deferred;

    std::vector<std::future<BlockDecodeResult>> blockTasks;
    blockTasks.reserve(block_count);

    for (uint32_t idx = 0; idx < block_count; ++idx) {
        uint32_t size_entry = block_sizes[idx];
        blockTasks.emplace_back(std::async(
            launch_policy,
            [this, &br, size_entry, idx, isStereo, useMidSide, &br_mutex, &br_cv, &next_block]() -> BlockDecodeResult {
                if (this->collector) {
                    this->collector->record(std::this_thread::get_id());
                }
                BlockDecodeResult result;
                result.index = idx;
                Block::Decoder blockDec;
                std::vector<int32_t> blockA;
                std::vector<int32_t> blockB;

                {
                    std::unique_lock<std::mutex> lock(br_mutex);
                    br_cv.wait(lock, [&] { return next_block == idx; });

                    if (!blockDec.decode(br, size_entry, blockA)) {
                        result.ok = false;
                    } else if (isStereo) {
                        if (!blockDec.decode(br, size_entry, blockB)) {
                            result.ok = false;
                        }
                    }

                    next_block++;
                    lock.unlock();
                    br_cv.notify_all();
                }

                if (!result.ok) {
                    return result;
                }

                if (!isStereo) {
                    result.left = std::move(blockA);
                } else if (!useMidSide) {
                    result.left = std::move(blockA);
                    result.right = std::move(blockB);
                } else {
                    size_t n = blockA.size();
                    result.left.resize(n);
                    result.right.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        int32_t m = blockA[i];
                        int32_t s = blockB[i];
                        int32_t l = m + ((s + (s & 1)) >> 1);
                        result.left[i] = l;
                        result.right[i] = l - s;
                    }
                }

                return result;
            }
        ));
    }

    for (auto& futureResult : blockTasks) {
        BlockDecodeResult result = futureResult.get();
        if (!result.ok) {
            left.clear();
            right.clear();
            return false;
        }
        size_t offset = blockOffsets[result.index];
        std::copy(result.left.begin(), result.left.end(), left.begin() + offset);
        if (isStereo) {
            std::copy(result.right.begin(), result.right.end(), right.begin() + offset);
        }
    }

    if (br.has_error()) {
        left.clear();
        right.clear();
        return false;
    }

    if (hdr.channels == 2 && right.size() != left.size()) {
        left.clear();
        right.clear();
        return false;
    }

    if (out_header) {
        *out_header = hdr;
    }
    return true;
}

} // namespace LAC
