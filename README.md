# Lossless Audio Codec (LAC)

[![CI](https://github.com/audexdev/Lossless-Audio-Codec/actions/workflows/ci.yml/badge.svg)](https://github.com/audexdev/Lossless-Audio-Codec/actions/workflows/ci.yml)

LAC is an experimental C++20 lossless audio codec for PCM WAV audio. It is a compact implementation of a custom `.lac` container and bitstream with LPC prediction, adaptive Rice coding, mid/side stereo, zero-run residual coding, residual partitioning, Apple Silicon NEON acceleration, and multithreaded block encoding.

The project is intended for codec experimentation, implementation study, and reproducible work on lossless audio compression internals. The file format is still experimental and should not yet be treated as a long-term archival format.

## Features

- 16-bit and 24-bit PCM WAV input
- 44.1, 48, 96, and 192 kHz sample rates
- Mono and stereo audio
- LR, mid/side, and per-block stereo mode selection
- Fixed, FIR, and LPC predictors
- Adaptive Rice residual coding
- Zero-run and small-residual bin modes
- Block partitioning for local residual adaptation
- Apple Silicon NEON paths where available
- Multithreaded block encoding
- CLI roundtrip workflow: WAV -> LAC -> WAV

## Status

Current release: `v1.4.0`

The codec currently prioritizes correctness, readability, and experimentation over a frozen public API. Known hardening work remains around malformed input handling, fuzzing, large-file limits, and public packaging. See [SECURITY.md](SECURITY.md) and [docs/format.md](docs/format.md) for the current contract and risk areas.

## Build

Requirements:

- CMake 3.16+
- C++20 compiler
- pthread-compatible threading support

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The CLI will be available at:

```sh
./build/lac_cli
```

## Usage

Encode a WAV file:

```sh
./build/lac_cli encode input.wav output.lac
```

Decode a LAC file:

```sh
./build/lac_cli decode output.lac restored.wav
```

Choose a stereo mode explicitly:

```sh
./build/lac_cli encode input.wav output.lac --stereo-mode=lr
./build/lac_cli encode input.wav output.lac --stereo-mode=ms
```

Run the built-in synthetic self-test:

```sh
./build/lac_cli selftest
```

## Tests

Build and run the CTest suite:

```sh
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Debug -DLAC_BUILD_TESTS=ON
cmake --build build-tests --parallel
ctest --test-dir build-tests --output-on-failure
```

If the local `assets/` directory is present, the E2E tests also run the larger WAV fixtures in that directory. In clean checkouts and CI, the test suite falls back to generated WAV fixtures so the repository remains self-contained.

## Format

The current `.lac` bitstream is documented in [docs/format.md](docs/format.md). The format is versioned internally as frame version `2`, but it is not yet frozen for external compatibility.

## Security

Malformed audio/container inputs are security-relevant because decoders and parsers handle attacker-controlled sizes and bitstreams. Please read [SECURITY.md](SECURITY.md) before reporting crashes, hangs, or memory safety issues.

## Maintainer Workflow

Codec correctness, fuzzing readiness, security review, and release review are tracked in the [maintainer workflow](docs/maintainer-workflow.md).

## Repository Hygiene

Generated build directories, local audio fixtures, editor state, and large temporary outputs are intentionally ignored. The source tree is designed to build from tracked files alone.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
