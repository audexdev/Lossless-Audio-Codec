#include "decoder.hpp"

namespace LAC {

Decoder::Decoder() {}

bool Decoder::decode(const uint8_t* data, size_t size, std::vector<int32_t>& left, std::vector<int32_t>& right, FrameHeader* out_header) {
    left.clear();
    right.clear();

    FrameHeader hdr;
    size_t header_bytes = 0;
    if (!FrameHeader::parse(data, size, hdr, header_bytes))
        return false;

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
                    return false;
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
                    return false;
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
            return false;
        }
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
