# Fuzzing Roadmap

Fuzzing is part of the parser and decoder hardening workflow. The current goal is to make malformed-input testing routine without turning every CI run into a long fuzzing job.

## Initial Targets

Planned fuzz targets:

- WAV parser: arbitrary RIFF/WAVE byte streams
- LAC container decoder: `LAC::Decoder::decode`
- block decoder: `Block::Decoder::decode`
- Rice and bitstream primitives
- zero-run and small-residual residual modes

## Build Strategy

The intended first implementation is libFuzzer-compatible and sanitizer-backed:

```sh
cmake -S . -B build-fuzz -DCMAKE_BUILD_TYPE=Debug -DLAC_BUILD_TESTS=ON
cmake --build build-fuzz --parallel
```

The exact CMake options will be documented when fuzz targets land. ASan/UBSan should be enabled for local fuzz runs and short CI smoke runs.

## Corpus Strategy

Seed corpora should start small:

- a tiny valid mono `.lac` stream
- a tiny valid stereo `.lac` stream
- a tiny valid 16-bit WAV
- a tiny valid 24-bit WAV
- minimized malformed inputs from fixed bugs, when safe to publish

Sensitive or exploitable crash inputs should follow `SECURITY.md` and should not be attached to public issues unless they are safe to disclose.

## CI Strategy

CI should start with short smoke runs that prove the targets build and can execute briefly. Longer fuzzing can run locally or on a scheduled workflow once runtime and corpus size are predictable.

## Tracking

The active tracking issue is #12.
