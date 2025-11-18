#include <iostream>
#include <vector>
#include "codec/bitstream/bit_writer.hpp"
#include "codec/bitstream/bit_reader.hpp"
#include "codec/rice/rice.hpp"

int main() {
    std::vector<int32_t> samples = {
        0, 1, -1, 5, -5, 100, -100,
        12345, -12345,
        32767, -32768,
        42, -42
    };

    uint32_t k = 4;

    BitWriter writer;
    for (int32_t v : samples) {
        Rice::encode(writer, v, k);
    }
    writer.flush_to_byte();

    const std::vector<uint8_t>& buf = writer.get_buffer();

    BitReader reader(buf);

    bool ok = true;
    for (size_t i = 0; i < samples.size(); ++i) {
        int32_t decoded = Rice::decode(reader, k);
        if (decoded != samples[i]) {
            std::cout << "Mismatch at index " << i
                      << " expected=" << samples[i]
                      << " got=" << decoded << "\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "RICE TEST PASSED\n";
    }
}