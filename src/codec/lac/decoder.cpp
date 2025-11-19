#include "decoder.hpp"

namespace LAC {

Decoder::Decoder() {}

void Decoder::decode(const uint8_t* data, size_t size, std::vector<int32_t>& left, std::vector<int32_t>& right) {
    BitReader br(data, size);

    FrameHeader hdr;
    hdr.read(br);

    if (hdr.sync != 0x4C41 || hdr.version != 1 ||
        (hdr.channels != 1 && hdr.channels != 2) ||
        hdr.sample_rate != 44100 || hdr.bit_depth != 16 ||
        hdr.block_size == 0 || hdr.reserved != 0)
        return;

    Block::Decoder blockDec;

    left.clear();
    right.clear();

    while (!br.eof()) {
        std::vector<int32_t> pcmL = blockDec.decode(br);
        if (pcmL.empty()) break;
        left.insert(left.end(), pcmL.begin(), pcmL.end());

        if (hdr.channels == 2) {
            std::vector<int32_t> pcmR = blockDec.decode(br);
            if (pcmR.empty()) break;
            right.insert(right.end(), pcmR.begin(), pcmR.end());
        }
    }

    if (!right.empty() && right.size() != left.size()) {
        right.clear();
        left.clear();
    }
}

} // namespace LAC
