# Changelog

All notable user-facing changes should be documented here. LAC is still experimental, so format and API compatibility changes must be called out explicitly.

## Unreleased

- Added public maintainer workflow, contributor guidance, issue templates, PR checklist, support policy, and release checklist.
- Added CI coverage for Debug tests, Release builds, and ASan/UBSan smoke tests on GitHub Actions.
- Added self-contained generated WAV fixtures for clean-checkout CI test runs.
- Updated E2E tests to read back temporary `.lac` files before decode.
- Removed tracked editor cache files and generated compile database symlink from source control.
- Expanded repository roadmap tracking for correctness, fuzzing, security hardening, and release readiness.

## v1.4.0

- Current public release.
- Experimental `.lac` format version 2.
- Supports PCM WAV roundtrips for the documented 16-bit and 24-bit mono/stereo domain.
- Includes LPC, fixed, and FIR predictors; adaptive Rice coding; zero-run and small-residual modes; block partitioning; mid/side stereo; NEON paths where available; and multithreaded block encoding.
