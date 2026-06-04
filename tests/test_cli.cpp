#include "codec/block/constants.hpp"
#include "io/wav_io.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

std::string shell_quote(const std::string& value) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
#else
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out + "'";
#endif
}

bool run_cli(const std::filesystem::path& cli, const std::vector<std::string>& args) {
    std::string command = shell_quote(cli.string());
    for (const auto& arg : args) {
        command += " " + shell_quote(arg);
    }
#ifdef _WIN32
    command += " > NUL 2>&1";
#else
    command += " > /dev/null 2>&1";
#endif
    return std::system(command.c_str()) == 0;
}

std::vector<uint8_t> load_binary_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f.good());
    f.seekg(0, std::ios::end);
    const std::streampos end = f.tellg();
    assert(end >= 0);
    std::vector<uint8_t> data(static_cast<size_t>(end));
    f.seekg(0, std::ios::beg);
    if (!data.empty()) {
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        assert(f.good());
    }
    return data;
}

void save_binary_file(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    assert(f.good());
    if (!data.empty()) {
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    f.close();
    assert(f.good());
}

void assert_no_staged_output_siblings(const std::filesystem::path& dir) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        assert(entry.path().filename().string().find(".lac-tmp.") == std::string::npos);
    }
}

uint16_t read_u16_le(const std::vector<uint8_t>& data, size_t offset) {
    assert(offset + 2 <= data.size());
    return static_cast<uint16_t>(
        static_cast<uint16_t>(data[offset]) |
        (static_cast<uint16_t>(data[offset + 1]) << 8));
}

uint32_t read_u32_le(const std::vector<uint8_t>& data, size_t offset) {
    assert(offset + 4 <= data.size());
    return static_cast<uint32_t>(
        static_cast<uint32_t>(data[offset]) |
        (static_cast<uint32_t>(data[offset + 1]) << 8) |
        (static_cast<uint32_t>(data[offset + 2]) << 16) |
        (static_cast<uint32_t>(data[offset + 3]) << 24));
}

void assert_fourcc(const std::vector<uint8_t>& data, size_t offset, const char* expected) {
    assert(offset + 4 <= data.size());
    assert(std::equal(data.begin() + static_cast<std::ptrdiff_t>(offset),
                      data.begin() + static_cast<std::ptrdiff_t>(offset + 4),
                      expected));
}

void assert_canonical_wav_layout(const std::filesystem::path& path,
                                 uint16_t channels,
                                 uint32_t sample_rate,
                                 uint8_t bit_depth,
                                 size_t frames) {
    const auto data = load_binary_file(path);
    const uint32_t block_align = channels * (bit_depth / 8u);
    const uint32_t payload_bytes = static_cast<uint32_t>(frames * block_align);
    const uint32_t padding_bytes = payload_bytes & 1u;
    assert(data.size() == 44u + payload_bytes + padding_bytes);
    assert_fourcc(data, 0, "RIFF");
    assert(read_u32_le(data, 4) == 36u + payload_bytes + padding_bytes);
    assert_fourcc(data, 8, "WAVE");
    assert_fourcc(data, 12, "fmt ");
    assert(read_u32_le(data, 16) == 16u);
    assert(read_u16_le(data, 20) == 1u);
    assert(read_u16_le(data, 22) == channels);
    assert(read_u32_le(data, 24) == sample_rate);
    assert(read_u32_le(data, 28) == sample_rate * block_align);
    assert(read_u16_le(data, 32) == block_align);
    assert(read_u16_le(data, 34) == bit_depth);
    assert_fourcc(data, 36, "data");
    assert(read_u32_le(data, 40) == payload_bytes);
    if (padding_bytes != 0u) {
        assert(data.back() == 0u);
    }
}

std::vector<int32_t> make_samples(size_t frames, uint8_t bit_depth, int phase) {
    assert(frames >= 16);
    std::vector<int32_t> out(frames);
    const int32_t min_sample = (bit_depth == 24) ? -0x800000 : -32768;
    const int32_t max_sample = (bit_depth == 24) ? 0x7FFFFF : 32767;
    for (size_t i = 0; i < frames; ++i) {
        out[i] = static_cast<int32_t>((i * 97u + static_cast<size_t>(phase)) % 20001u) - 10000;
    }
    out[0] = min_sample;
    out[1] = max_sample;
    for (size_t i = 2; i < 16; ++i) out[i] = 0;
    return out;
}

void assert_roundtrip(const std::filesystem::path& cli,
                      const std::filesystem::path& dir,
                      const std::string& label,
                      uint16_t channels,
                      uint32_t sample_rate,
                      uint8_t bit_depth,
                      const std::vector<std::string>& encode_flags = {},
                      size_t frames = 1024) {
    const auto input = dir / (label + ".wav");
    const auto encoded = dir / (label + ".lac");
    const auto restored = dir / (label + "_restored.wav");
    const auto left = make_samples(frames, bit_depth, 0);
    const auto right = (channels == 2) ? make_samples(frames, bit_depth, 37) : std::vector<int32_t>{};

    assert(write_wav(input.string(), left, right, channels, sample_rate, bit_depth));
    std::vector<std::string> encode_args = {"encode", input.string(), encoded.string()};
    encode_args.insert(encode_args.end(), encode_flags.begin(), encode_flags.end());
    assert(run_cli(cli, encode_args));
    assert(run_cli(cli, {"decode", encoded.string(), restored.string(), "--threads=2"}));

    std::vector<int32_t> restored_left;
    std::vector<int32_t> restored_right;
    uint16_t restored_channels = 0;
    uint32_t restored_rate = 0;
    uint8_t restored_depth = 0;
    assert(read_wav(restored.string(),
                    restored_left,
                    restored_right,
                    restored_channels,
                    restored_rate,
                    restored_depth));
    assert(restored_left == left);
    assert(restored_right == right);
    assert(restored_channels == channels);
    assert(restored_rate == sample_rate);
    assert(restored_depth == bit_depth);
    assert_canonical_wav_layout(restored, channels, sample_rate, bit_depth, frames);
}

void assert_same_path_rejected(const std::filesystem::path& cli,
                               const std::filesystem::path& dir) {
    const auto wav = dir / "same_path.wav";
    const auto lac = dir / "same_path.lac";
    const auto left = make_samples(64, 16, 0);
    assert(write_wav(wav.string(), left, {}, 1, 44100, 16));
    const auto wav_before = load_binary_file(wav);
    assert(!run_cli(cli, {"encode", wav.string(), wav.string()}));
    assert(load_binary_file(wav) == wav_before);

    assert(run_cli(cli, {"encode", wav.string(), lac.string()}));
    const auto lac_before = load_binary_file(lac);
    assert(!run_cli(cli, {"decode", lac.string(), lac.string()}));
    assert(load_binary_file(lac) == lac_before);

    const auto wav_alias = dir / "same_path_alias.wav";
    std::error_code ec;
    std::filesystem::create_hard_link(wav, wav_alias, ec);
    if (!ec) {
        assert(!run_cli(cli, {"encode", wav.string(), wav_alias.string()}));
        assert(load_binary_file(wav) == wav_before);
    }

    const auto lac_alias = dir / "same_path_alias.lac";
    ec.clear();
    std::filesystem::create_hard_link(lac, lac_alias, ec);
    if (!ec) {
        assert(!run_cli(cli, {"decode", lac.string(), lac_alias.string()}));
        assert(load_binary_file(lac) == lac_before);
    }
    assert_no_staged_output_siblings(dir);
}

void assert_malformed_input_rejected_without_clobber(const std::filesystem::path& cli,
                                                     const std::filesystem::path& dir) {
    const auto valid_wav = dir / "malformed_source.wav";
    const auto malformed_wav = dir / "malformed.wav";
    const auto lac_output = dir / "malformed_output.lac";
    const auto valid_lac = dir / "valid.lac";
    const auto malformed_lac = dir / "malformed.lac";
    const auto wav_output = dir / "malformed_output.wav";
    assert(write_wav(valid_wav.string(), make_samples(64, 16, 0), {}, 1, 44100, 16));

    auto wav_bytes = load_binary_file(valid_wav);
    wav_bytes[4] = 0;
    save_binary_file(malformed_wav, wav_bytes);
    const std::vector<uint8_t> lac_sentinel = {0xAAu, 0x55u, 0xAAu};
    save_binary_file(lac_output, lac_sentinel);
    assert(!run_cli(cli, {"encode", malformed_wav.string(), lac_output.string()}));
    assert(load_binary_file(lac_output) == lac_sentinel);
    assert_no_staged_output_siblings(dir);

    assert(run_cli(cli, {"encode", valid_wav.string(), valid_lac.string()}));
    auto lac_bytes = load_binary_file(valid_lac);
    lac_bytes.push_back(0);
    save_binary_file(malformed_lac, lac_bytes);
    const std::vector<uint8_t> wav_sentinel = {0x52u, 0x49u, 0x46u, 0x46u};
    save_binary_file(wav_output, wav_sentinel);
    assert(!run_cli(cli, {"decode", malformed_lac.string(), wav_output.string()}));
    assert(load_binary_file(wav_output) == wav_sentinel);
    assert_no_staged_output_siblings(dir);
}

void assert_existing_output_overwritten(const std::filesystem::path& cli,
                                        const std::filesystem::path& dir) {
    const auto wav = dir / "overwrite_source.wav";
    const auto lac = dir / "overwrite_output.lac";
    const auto restored = dir / "overwrite_restored.wav";
    const auto left = make_samples(64, 24, 0);
    const auto right = make_samples(64, 24, 37);
    assert(write_wav(wav.string(), left, right, 2, 48000, 24));

    const std::vector<uint8_t> sentinel = {0xAAu, 0x55u, 0xAAu};
    save_binary_file(lac, sentinel);
    assert(run_cli(cli, {"encode", wav.string(), lac.string()}));
    assert(load_binary_file(lac) != sentinel);
    assert_no_staged_output_siblings(dir);

    save_binary_file(restored, sentinel);
    assert(run_cli(cli, {"decode", lac.string(), restored.string()}));
    std::vector<int32_t> restored_left;
    std::vector<int32_t> restored_right;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint8_t bit_depth = 0;
    assert(read_wav(restored.string(),
                    restored_left,
                    restored_right,
                    channels,
                    sample_rate,
                    bit_depth));
    assert(restored_left == left);
    assert(restored_right == right);
    assert(channels == 2);
    assert(sample_rate == 48000);
    assert(bit_depth == 24);
    assert_no_staged_output_siblings(dir);
}

void assert_publish_failure_cleans_up(const std::filesystem::path& cli,
                                      const std::filesystem::path& dir) {
    const auto wav = dir / "publish_failure_source.wav";
    const auto lac = dir / "publish_failure_source.lac";
    const auto output_dir = dir / "publish_failure_output";
    const auto marker = output_dir / "marker";
    assert(write_wav(wav.string(), make_samples(64, 16, 0), {}, 1, 44100, 16));
    assert(run_cli(cli, {"encode", wav.string(), lac.string()}));
    assert(std::filesystem::create_directory(output_dir));
    const std::vector<uint8_t> sentinel = {0x11u, 0x22u, 0x33u};
    save_binary_file(marker, sentinel);

    assert(!run_cli(cli, {"encode", wav.string(), output_dir.string()}));
    assert(load_binary_file(marker) == sentinel);
    assert_no_staged_output_siblings(dir);

    assert(!run_cli(cli, {"decode", lac.string(), output_dir.string()}));
    assert(load_binary_file(marker) == sentinel);
    assert_no_staged_output_siblings(dir);
}

void assert_link_targets_not_clobbered(const std::filesystem::path& cli,
                                       const std::filesystem::path& dir) {
    const auto wav = dir / "link_source.wav";
    const auto baseline_lac = dir / "link_source.lac";
    assert(write_wav(wav.string(), make_samples(64, 16, 0), {}, 1, 44100, 16));
    assert(run_cli(cli, {"encode", wav.string(), baseline_lac.string()}));

    const std::vector<uint8_t> sentinel = {0x44u, 0x55u, 0x66u};
    const auto hardlink_target = dir / "hardlink_target";
    const auto hardlink_output = dir / "hardlink_output.lac";
    save_binary_file(hardlink_target, sentinel);
    std::error_code ec;
    std::filesystem::create_hard_link(hardlink_target, hardlink_output, ec);
    if (!ec) {
        assert(run_cli(cli, {"encode", wav.string(), hardlink_output.string()}));
        assert(load_binary_file(hardlink_target) == sentinel);
        assert(load_binary_file(hardlink_output) != sentinel);
    }
    assert_no_staged_output_siblings(dir);

    const auto symlink_target = dir / "symlink_target";
    const auto symlink_output = dir / "symlink_output.wav";
    save_binary_file(symlink_target, sentinel);
    ec.clear();
    std::filesystem::create_symlink(symlink_target, symlink_output, ec);
    if (!ec) {
        assert(run_cli(cli, {"decode", baseline_lac.string(), symlink_output.string()}));
        assert(load_binary_file(symlink_target) == sentinel);
        assert(!std::filesystem::is_symlink(symlink_output));
    }
    assert_no_staged_output_siblings(dir);
}

void assert_long_output_filenames_supported(const std::filesystem::path& cli,
                                            const std::filesystem::path& dir) {
    const auto wav = dir / "long_name_source.wav";
    const auto lac = dir / (std::string(240, 'l') + ".lac");
    const auto restored = dir / (std::string(240, 'w') + ".wav");
    assert(write_wav(wav.string(), make_samples(64, 16, 0), {}, 1, 44100, 16));
    assert(run_cli(cli, {"encode", wav.string(), lac.string()}));
    assert(run_cli(cli, {"decode", lac.string(), restored.string()}));

    std::vector<int32_t> restored_left;
    std::vector<int32_t> restored_right;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint8_t bit_depth = 0;
    assert(read_wav(restored.string(),
                    restored_left,
                    restored_right,
                    channels,
                    sample_rate,
                    bit_depth));
    assert(restored_left == make_samples(64, 16, 0));
    assert(restored_right.empty());
    assert_no_staged_output_siblings(dir);
}

void assert_restrictive_umask_supported(const std::filesystem::path& cli,
                                        const std::filesystem::path& dir) {
#ifndef _WIN32
    const auto wav = dir / "umask_source.wav";
    const auto lac = dir / "umask_output.lac";
    const auto restored = dir / "umask_restored.wav";
    assert(write_wav(wav.string(), make_samples(64, 16, 0), {}, 1, 44100, 16));

    const mode_t previous_encode_umask = ::umask(0777);
    const bool encoded = run_cli(cli, {"encode", wav.string(), lac.string()});
    ::umask(previous_encode_umask);
    assert(encoded);

    std::error_code ec;
    std::filesystem::permissions(
        lac,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace,
        ec);
    assert(!ec);

    const mode_t previous_decode_umask = ::umask(0777);
    const bool decoded = run_cli(cli, {"decode", lac.string(), restored.string()});
    ::umask(previous_decode_umask);
    assert(decoded);

    std::filesystem::permissions(
        restored,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace,
        ec);
    assert(!ec);
    assert_no_staged_output_siblings(dir);
#else
    (void)cli;
    (void)dir;
#endif
}

} // namespace

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::filesystem::path cli = argv[1];
    assert(std::filesystem::exists(cli));

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("lac_cli_tests_" + std::to_string(unique));
    assert(std::filesystem::create_directories(dir));

    for (uint16_t channels : {1u, 2u}) {
        for (uint32_t sample_rate : {44100u, 48000u, 96000u, 192000u}) {
            for (uint8_t bit_depth : {16u, 24u}) {
                const std::string label =
                    std::to_string(channels) + "ch_" +
                    std::to_string(bit_depth) + "_" + std::to_string(sample_rate);
                assert_roundtrip(cli, dir, label, channels, sample_rate, bit_depth);
            }
        }
    }
    assert_roundtrip(cli, dir, "stereo_24_48000_lr", 2, 48000, 24, {"--stereo-mode=lr"});
    assert_roundtrip(cli, dir, "stereo_24_48000_ms", 2, 48000, 24, {"--stereo-mode=ms"});
    assert_roundtrip(cli, dir, "mono_24_44100_odd", 1, 44100, 24, {}, 17);
    assert_roundtrip(cli, dir, "stereo_16_44100_multiblock", 2, 44100, 16, {},
                     Block::MAX_BLOCK_SIZE + 37u);
    assert_same_path_rejected(cli, dir);
    assert_malformed_input_rejected_without_clobber(cli, dir);
    assert_existing_output_overwritten(cli, dir);
    assert_publish_failure_cleans_up(cli, dir);
    assert_link_targets_not_clobbered(cli, dir);
    assert_long_output_filenames_supported(cli, dir);
    assert_restrictive_umask_supported(cli, dir);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::cout << "cli roundtrip tests ok\n";
    return 0;
}
