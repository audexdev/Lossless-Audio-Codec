#!/bin/bash
set -e

cd "$(dirname "$0")/build-release"

if command -v gdate >/dev/null 2>&1; then
    DATE_CMD="gdate"
else
    DATE_CMD="date"
fi

measure() {
    label="$1"
    shift

    echo ">>> $label"

    start_ms=$($DATE_CMD +%s%3N)
    "$@"
    end_ms=$($DATE_CMD +%s%3N)

    elapsed_ms=$((end_ms - start_ms))
    echo "  time=${elapsed_ms}ms"
    echo ""
}

# Encode
measure "encode 16.44100.wav (LR)" ./lac_cli encode ../assets/16.44100.wav 16.44100.lac

measure "encode 24.44100.wav (LR)" ./lac_cli encode ../assets/24.44100.wav 24.44100.lac

measure "encode 24.48000.wav (LR)" ./lac_cli encode ../assets/24.48000.wav 24.48000.lac

measure "encode 24.96000.wav (LR)" ./lac_cli encode ../assets/24.96000.wav 24.96000.lac

measure "encode 24.192000.wav (LR)" ./lac_cli encode ../assets/24.192000.wav 24.192000.lac

# Decode
measure "decode 16.44100.lac" ./lac_cli decode 16.44100.lac r_16.44100.wav

measure "decode 24.44100.lac" ./lac_cli decode 24.44100.lac r_24.44100.wav

measure "decode 24.48000.lac" ./lac_cli decode 24.48000.lac r_24.48000.wav

measure "decode 24.96000.lac" ./lac_cli decode 24.96000.lac r_24.96000.wav

measure "decode 24.192000.lac" ./lac_cli decode 24.192000.lac r_24.192000.wav

echo "--------------------------------------"
echo " All encode/decode operations complete "
echo "--------------------------------------"