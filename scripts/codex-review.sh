#!/usr/bin/env bash
#
# Cross-model PR review for LAC using the Codex CLI (non-interactive / one-shot).
# The idea: Claude implements + verifies, an independent model (Codex) reviews,
# so the two models do not share the same blind spots.
#
# Usage:
#   scripts/codex-review.sh          # review the current branch vs origin/main
#   scripts/codex-review.sh 23       # review GitHub PR #23
#
# Notes:
#   - No model is pinned: this uses the Codex CLI default model.
#   - Reasoning effort is raised via EFFORT_ARGS below. If the config key differs
#     in your Codex version, adjust it there (check: codex exec --help). Set
#     EFFORT_ARGS=() to fall back to Codex defaults.
#   - Codex runs the diff command itself and reads the touched files for context,
#     so nothing is piped in. The review is read-only.
set -euo pipefail

# --- tunable knobs (kept in one place on purpose) ---------------------------
EFFORT_ARGS=(-c model_reasoning_effort="high")
# ---------------------------------------------------------------------------

if ! command -v codex >/dev/null 2>&1; then
  echo "error: 'codex' CLI not found on PATH" >&2
  exit 1
fi

if [[ $# -ge 1 ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "error: reviewing a PR needs the 'gh' CLI" >&2
    exit 1
  fi
  diff_cmd="gh pr diff $1"
  target="GitHub PR #$1"
else
  diff_cmd="git diff origin/main...HEAD"
  target="the current branch vs origin/main"
fi

read -r -d '' prompt <<EOF || true
Read-only review. Run: ${diff_cmd}
You are reviewing a change to the LAC lossless audio codec (${target}).
The hard invariant is a bit-perfect round-trip: WAV -> encode -> .lac -> decode
-> WAV must be identical for the supported domain (16/24-bit, 1-2 channels,
44.1/48/96/192 kHz). Read the touched files for context before judging.

Check specifically:
1. Could this break the lossless round-trip for any input?
2. If it touches predictor / Rice / zigzag / partition logic on ONE side
   (encoder OR decoder), is the other side updated to match? They are
   hand-synced duplicate implementations; a one-sided change silently corrupts.
3. SIMD (src/codec/simd/neon.cpp) must be bit-identical to the scalar path.
4. Treat WAV/.lac as attacker-controlled: bound allocations, reject malformed
   input cleanly, no undefined behaviour on full-range int32 values.
5. C++20 portability: macOS libc++ lacks some features (e.g. std::jthread);
   there is no x86 SIMD path.
6. If the bitstream changed, is docs/format.md updated to match?

List concrete findings as file:line + severity (high/medium/low). Be skeptical.
Do not modify any files.
EOF

exec codex exec "${EFFORT_ARGS[@]}" "$prompt"
