# Refactor Roadmap

This document sequences the design-debt work tracked in issues #23–#30 against the
existing hardening, format, and performance issues. LAC is experimental, but these
changes are correctness work: the lossless round-trip
(`supported WAV -> encode -> .lac -> decode -> restored WAV`) must hold at every
step, not only at the end.

The ordering is dependency-driven, not severity-driven. Several high-value items
(#23, #24, #26) carry the highest risk of silently breaking losslessness, so they
are deliberately placed *after* the test-strengthening and undefined-behavior work
that protects them.

## Guiding Principles

- The bit-perfect round-trip and the malformed-input rejection behavior are
  invariants. No step may regress either; each step lands with the round-trip and
  malformed-input tests green.
- Strengthen the test safety net before risky de-duplication.
- Establish a single source of truth for a contract before restructuring the code
  that depends on it.
- Group changes that touch the same files (the codec arithmetic hot paths) so the
  same code is not rewritten twice.
- Prefer no observable change to emitted/accepted bytes during pure refactors;
  protect that with fixtures that must still round-trip bit-for-bit.

## Sequenced Plan

### Phase 1 — Foundation and test safety net

Low-risk work that removes noise and makes everything afterwards easier to test.

1. **#30 — Remove dead code and empty placeholders; move `selftest` into the test
   suite.** Deleting `Rice::compute_k`, the unused `Block::Encoder` order member,
   and the empty placeholder files reduces confusion before refactoring. Moving
   `lac_cli selftest` into the test suite and extending it to cover per-block
   stereo (mode 2) and mono strengthens the regression net used by every later
   phase.
2. **#27 — Remove hidden environment coupling from the library; unify
   configuration.** Once the codec no longer reads `LAC_THREADS` / `LAC_DEBUG_*`
   from inside `encode`/`decode`, encoder and decoder become deterministic to
   drive from tests, which the later phases depend on.

Gate: round-trip, predictor/residual, partitioning, and zero-run tests all pass;
`selftest` coverage now includes mode 2 and mono.

### Phase 2 — Correct and unify the core arithmetic

These items all touch the predictor / Rice / LPC hot paths, so they are done as one
cluster to avoid rewriting the same code repeatedly.

3. **#8 (existing) — Eliminate undefined / implementation-defined arithmetic.**
   Fix the UB and unchecked narrowing first, so the code that is about to be
   unified is already correct.
4. **#23 — Define a single source of truth for the format contract.** Extract the
   predictor formulas, zigzag mapping, stateless `adapt_k`, partition-size math,
   the Q15 shift/scale, and the wire-format tag/mode constants into shared
   definitions consumed by encoder, decoder, and SIMD. This is the foundation for
   #24, #25, and #26.
5. **#26 — Guarantee SIMD/scalar bit-exactness; unify NEON LPC range-safety.**
   With the arithmetic unified (#23) and UB removed (#8), collapse the three
   residual engines onto one shared kernel, add a test asserting NEON output is
   byte-identical to the scalar reference, and move range-checking into the path
   that writes residuals.

Gate: a SIMD-vs-scalar equality test passes on representative and near-overflow
inputs; sanitizer builds are clean on the codec hot paths.

### Phase 3 — Consolidate the decode path

4. **#24 — Consolidate the duplicated v3 decode path, validation helpers, resource
   limits, and WAV writer.** Done together with **#3 (bound decoder allocations)**
   and **#7 (reject non-canonical metadata)**, since all three touch the same
   parser surface. This collapses the second attacker-facing parser in
   `main.cpp` onto the shared one and reuses the validators unified in #23.
5. **#28 — Adopt one error-handling strategy across encode, decode, I/O, and
   CLI.** Apply it as the decode path is consolidated, so the convention is
   settled in one place and aligns with #7's decode-time rejection requirements.

Gate: a single v3 block-table parser is exercised by both the CLI fast path and
`LAC::Decoder`; malformed-input tests assert on specific rejection reasons; the
resource-limit constants exist in exactly one place.

### Phase 4 — Structure and platform

6. **#25 — Decompose the monolithic block/frame encode functions.** With the
   formulas extracted in #23, `Block::Encoder::encode` shrinks; separating
   plan/select/serialize makes the emit routines unit-testable and unblocks
   **#15 (reduce peak memory and repeated analysis)**.
7. **#29 — Address the x86 SIMD gap, misleading `neon_available()`, and unguarded
   endianness.** Platform expansion comes last. The compile-time endianness
   `static_assert` is a small, safe change that may be pulled forward at any time.

Gate: encode decomposition lands with no change to emitted bytes; platform
behavior (SIMD availability, supported endianness) is documented.

## Dependency Summary

- #30, #27 → enable reliable, deterministic testing for everything after.
- #8 → precedes #23/#26 (fix before restructure).
- #23 → unblocks #24 (shared validators), #25 (smaller encode), #26 (shared
  kernel).
- #24 → precedes/co-lands with #28 (settle the error convention on the
  consolidated path); pairs with #3 and #7.
- #25 → unblocks #15.
- #29 → independent; last.

## Cross-Cutting Work

- **#5 (multithreaded encoder startup exception-safety)** is an independent, small
  hardening item that can be slotted in at any time; it pairs naturally with the
  worker-pool unification in #24.
- **#11 (malformed-input regression tests)** and **#12 (fuzzing harnesses)** are
  ongoing. Landing them before or alongside Phase 3 increases confidence in the
  decode-path consolidation.

## Status

This is a living plan. Update it when an issue is closed, reordered, or when a
dependency assumption changes. Per `maintainer-workflow.md`, verified work should
still leave normal artifacts behind (issues, PRs with correctness/security notes,
CI runs, regression tests).
