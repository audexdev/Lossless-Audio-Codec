#include "decoder.hpp"

namespace LAC {

Decoder::Decoder() {}

void Decoder::decode(const uint8_t* data, size_t size, std::vector<int32_t>& left, std::vector<int32_t>& right) {
    left.clear();
    right.clear();

    FrameHeader hdr;
    size_t header_bytes = 0;
    if (!FrameHeader::parse(data, size, hdr, header_bytes))
        return;

    BitReader br(data + header_bytes, size - header_bytes);

    Block::Decoder blockDec;

    if (hdr.channels == 1) {
        while (!br.eof()) {
            std::vector<int32_t> pcmL = blockDec.decode(br);
            if (pcmL.empty()) break;
            left.insert(left.end(), pcmL.begin(), pcmL.end());
        }
    } else {
        if (hdr.stereo_mode == 0) {
            while (!br.eof()) {
                std::vector<int32_t> pcmL = blockDec.decode(br);
                if (pcmL.empty()) break;
                std::vector<int32_t> pcmR = blockDec.decode(br);
                if (pcmR.empty() || pcmR.size() != pcmL.size()) {
                    left.clear();
                    right.clear();
                    return;
                }
                left.insert(left.end(), pcmL.begin(), pcmL.end());
                right.insert(right.end(), pcmR.begin(), pcmR.end());
            }
        } else if (hdr.stereo_mode == 1) {
            while (!br.eof()) {
                std::vector<int32_t> mid = blockDec.decode(br);
                if (mid.empty()) break;
                std::vector<int32_t> side = blockDec.decode(br);
                if (side.empty() || side.size() != mid.size()) {
                    left.clear();
                    right.clear();
                    return;
                }
                size_t n = mid.size();
                size_t base = left.size();
                left.resize(base + n);
                right.resize(base + n);
                int32_t* lptr = left.data() + base;
                int32_t* rptr = right.data() + base;
                for (size_t i = 0; i < n; ++i) {
                    int32_t m = mid[i];
                    int32_t s = side[i];
                    int32_t l = m + ((s + (s & 1)) >> 1);
                    lptr[i] = l;
                    rptr[i] = l - s;
                }
            }
        } else {
            return;
        }
    }
}

} // namespace LAC
