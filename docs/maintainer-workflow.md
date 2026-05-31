# Maintainer Workflow

This document describes the routine maintenance workflow used for LAC. The project is experimental, but codec changes still need to be reviewed as correctness, parser hardening, and release-quality work rather than as isolated implementation changes.

## Review Goals

Maintainer review focuses on four areas:

- codec correctness: encoded streams must decode to the expected PCM and metadata for the supported WAV domain
- fuzzing readiness: parser and decoder surfaces should be easy to exercise with malformed inputs and sanitizer builds
- security review: attacker-controlled sizes, bitstreams, and file paths should fail safely without crashes, hangs, or excessive allocation
- release review: public docs, version metadata, CI, known risks, and release notes should agree before a release is tagged

## Codec Correctness

Correctness review starts from the format contract and then checks the implementation and tests against that contract.

Maintainers should verify:

- the supported WAV domain is explicit: channel count, sample rates, bit depth, sample range, and unsupported variants
- the `.lac` bitstream documentation is accurate enough to explain decoder behavior
- predictor, stereo, residual, and partitioning changes have targeted tests, not only broad roundtrip tests
- CLI and library roundtrips compare restored audio and metadata where practical
- changes that alter bitstream behavior update `docs/format.md` and note compatibility impact

Correctness findings should be tracked as issues or pull requests with enough detail to reproduce the case.

## Fuzzing

Fuzzing work is tracked as part of parser and decoder hardening. The main targets are:

- WAV parsing
- LAC container decoding
- block decoding
- Rice and bitstream primitives
- zero-run and small-residual modes

Short sanitizer-backed fuzz smoke tests are suitable for CI once the targets are stable. Longer fuzzing runs can remain local or scheduled until runtime and corpus size are predictable. Crashes found by fuzzing should be minimized, classified as security-relevant when appropriate, and converted into regression tests after the fix.

## Security Review

Security review treats malformed audio and container input as attacker-controlled data. The important failure modes are memory safety bugs, unchecked allocation, parser hangs, unsafe file handling, and acceptance of non-canonical metadata that can hide later decoder bugs.

Maintainers should check:

- declared lengths and counts are bounded before allocation
- malformed chunk and bitstream reads fail cleanly
- arithmetic on sample, residual, and predictor values avoids undefined behavior
- CLI input and output paths cannot accidentally clobber source files
- security-sensitive findings are reported according to `SECURITY.md`

Public issues should avoid attaching exploitable inputs or minimized crash files when private reporting is more appropriate.

## Release Review

Before tagging a release, maintainers should check:

- CI is green on the release commit
- README status, CMake version metadata, public headers, and release notes agree
- known correctness, security, and compatibility risks are either fixed or explicitly tracked
- the release does not introduce an undocumented format or API compatibility change
- build and test instructions work from a clean checkout

Release notes should call out user-visible behavior changes, format changes, security fixes, and known limitations.

## AI-Assisted Review

Maintainers may use Codex or other AI-assisted review tools to inspect the codebase, compare implementation behavior against the format documentation, draft issue checklists, and look for missing tests. These tools are used as review aids only.

Any AI-assisted finding still needs maintainer verification before it becomes project guidance. Verified findings should be recorded as normal GitHub issues or pull requests with concrete file references, reproduction notes, and acceptance criteria.

## Expected Outputs

Routine maintenance should leave normal project artifacts behind:

- issues for confirmed defects, hardening work, and roadmap items
- pull requests with correctness, security, fuzzing, release, and CI notes
- CI runs for pushed changes
- updated documentation when behavior or maintenance policy changes
- regression tests for fixed parser, decoder, and CLI failures
