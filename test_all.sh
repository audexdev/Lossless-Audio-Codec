#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/build-release"}"
ASSETS_DIR="${LAC_ASSETS_DIR:-"$ROOT_DIR/assets"}"
CLI="${LAC_CLI:-"$BUILD_DIR/lac_cli"}"

if [[ ! -x "$CLI" ]]; then
    echo "lac_cli not found or not executable: $CLI" >&2
    echo "Build first or set BUILD_DIR=/path/to/build." >&2
    exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lac_roundtrip.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

measure() {
    label="$1"
    shift

    echo ">>> $label"

    start_s=$(date +%s)
    "$@"
    end_s=$(date +%s)

    elapsed_s=$((end_s - start_s))
    echo "  time=${elapsed_s}s"
    echo ""
}

roundtrip() {
    filename="$1"
    src="$ASSETS_DIR/$filename"
    lac="$TMP_DIR/${filename%.wav}.lac"
    restored="$TMP_DIR/r_$filename"

    if [[ ! -f "$src" ]]; then
        echo "missing fixture: $src" >&2
        exit 1
    fi

    measure "encode $filename" "$CLI" encode "$src" "$lac"
    measure "decode ${filename%.wav}.lac" "$CLI" decode "$lac" "$restored"

    if ! cmp -s "$src" "$restored"; then
        echo "roundtrip mismatch: $filename" >&2
        exit 1
    fi

    echo "  verified=$filename"
    echo ""
}

roundtrip "16.44100.wav"
roundtrip "24.44100.wav"
roundtrip "24.48000.wav"
roundtrip "24.96000.wav"
roundtrip "24.192000.wav"

echo "--------------------------------------"
echo " All encode/decode roundtrips verified "
echo "--------------------------------------"
