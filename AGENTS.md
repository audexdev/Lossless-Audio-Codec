# Repository Instructions

## Project Direction

LAC is currently a CLI-first experimental lossless audio codec. Prioritize making
`lac_cli encode` and `lac_cli decode` reliable, bit-perfect, and well specified
for the documented PCM WAV domain before expanding the public library API.

The main correctness invariant is:

```text
supported WAV -> lac_cli encode -> .lac -> lac_cli decode -> restored WAV
```

The restored PCM samples and relevant WAV metadata must match the original for
supported inputs. Compression ratio improvements are secondary to correctness,
bounded resource use, and clear format behavior.

## Scope Preferences

- Keep changes small and aligned with the existing C++20/CMake structure.
- Do not redesign the project into a library-first package unless explicitly
  requested.
- Treat `docs/format.md` and `docs/supported-formats.md` as the source of truth
  for codec behavior. Update them with any bitstream, predictor, residual,
  stereo, validation, or compatibility change.
- Public-facing repository content should be written in English.
- GitHub issues are useful tracking artifacts, but they may be newer than the
  original project direction. Prefer local docs, release notes, tests, and code
  when inferring project intent.

## Testing And CPU Use

This project can use large local WAV fixtures under `assets/`. Do not run heavy
asset roundtrips by accident.

For normal development, prefer lightweight generated-fixture tests:

```sh
mkdir -p /tmp/empty-lac-assets
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Debug -DLAC_BUILD_TESTS=ON -DLAC_TEST_ASSETS_DIR=/tmp/empty-lac-assets
cmake --build build-tests --parallel 4
LAC_THREADS=4 ctest --test-dir build-tests --output-on-failure
```

For heavier CLI roundtrip verification with local `assets/`, cap encoder worker
threads. On this maintainer machine, use at most 12 encoder threads by default:

```sh
cmake --build build-release --parallel 12
LAC_THREADS=12 BUILD_DIR=build-release ./test_all.sh
```

Use lower values such as `LAC_THREADS=4` when running repeated checks during an
agent loop. The `test_all.sh` script defaults to `LAC_THREADS=12` when the
environment does not already set a limit.

## Correctness Work

When changing codec logic:

- Add or update targeted tests for predictor, residual, stereo, partitioning,
  bitstream, or CLI behavior.
- Preserve bit-perfect roundtrips for the supported WAV domain.
- Prefer deterministic generated fixtures for default tests.
- Use large local fixtures only for explicit release or heavy verification.
- If a behavior is currently experimental or intentionally unsupported, document
  it rather than relying on implicit implementation behavior.

## Security And Malformed Inputs

WAV and LAC inputs are attacker-controlled parser inputs. Crashes, hangs,
unchecked allocation, path clobbering, and acceptance of non-canonical metadata
are security-relevant even for local CLI workflows.

For parser or decoder hardening:

- Reject malformed input cleanly.
- Add regression tests for fixed malformed-input cases when practical.
- Do not publish exploit details or sensitive crashing inputs in public docs.
- Keep fuzzing hooks and corpora small enough for routine local use unless a
  heavy fuzzing run is explicitly requested.
