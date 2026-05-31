# Changelog

All notable user-facing changes should be documented here. LAC is still experimental, so format and API compatibility changes must be called out explicitly.

## Unreleased

- Added public maintainer workflow, contributor guidance, issue templates, PR checklist, support policy, and release checklist.
- Added CI coverage for Debug tests, Release builds, and ASan/UBSan smoke tests on GitHub Actions.
- Added self-contained generated WAV fixtures for clean-checkout CI test runs.
- Updated E2E tests to read back temporary `.lac` files before decode.
- Added encoder thread limiting through `LAC_THREADS` and `lac_cli encode --threads=N`.
- Clarified the CLI-first PCM WAV roundtrip contract and canonical restored-WAV behavior.
- Hardened WAV parsing against inconsistent RIFF sizes, malformed chunk boundaries, non-canonical PCM metadata, empty payloads, and unchecked data-chunk allocation.
- Hardened `.lac` decoding against non-canonical reserved fields, stereo flags, residual metadata, padding, trailing payload bytes, out-of-range restored samples, oversized decoded allocations, and malformed Rice values.
- Added generated `lac_cli` subprocess roundtrips, full supported WAV-domain generated fixtures, and malformed-input regression coverage.
- Rejected encode or decode commands whose output path refers to the input file.
- Fixed canonical RIFF padding for odd-sized restored PCM payloads and tightened close-time write error handling.
- Bounded tiny-block decoder work, made extreme zigzag decoding portable, and corrected the LPC reconstruction specification.
- Removed tracked editor cache files and generated compile database symlink from source control.
- Expanded repository roadmap tracking for correctness, fuzzing, security hardening, and release readiness.

## v1.4.0

- Current public release.
- Experimental `.lac` format version 2.
- Supports PCM WAV roundtrips for the documented 16-bit and 24-bit mono/stereo domain.
- Includes LPC, fixed, and FIR predictors; adaptive Rice coding; zero-run and small-residual modes; block partitioning; mid/side stereo; NEON paths where available; and multithreaded block encoding.
