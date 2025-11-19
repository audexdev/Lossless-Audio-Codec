#!/bin/bash

set -e

cd "$(dirname "$0")/build"

# Encode
./lac_cli encode ../assets/16.44100.wav 16.44100.lac --stereo-mode=ms
./lac_cli encode ../assets/16.44100.wav 16.44100_lr.lac

./lac_cli encode ../assets/24.44100.wav 24.44100.lac --stereo-mode=ms
./lac_cli encode ../assets/24.44100.wav 24.44100_lr.lac

./lac_cli encode ../assets/24.48000.wav 24.48000.lac --stereo-mode=ms
./lac_cli encode ../assets/24.48000.wav 24.48000_lr.lac

./lac_cli encode ../assets/24.96000.wav 24.96000.lac --stereo-mode=ms
./lac_cli encode ../assets/24.96000.wav 24.96000_lr.lac

./lac_cli encode ../assets/24.192000.wav 24.192000.lac --stereo-mode=ms
./lac_cli encode ../assets/24.192000.wav 24.192000_lr.lac

# Decode
./lac_cli decode 16.44100.lac r_16.44100.wav
./lac_cli decode 16.44100_lr.lac r_16.44100_lr.wav

./lac_cli decode 24.44100.lac r_24.44100.wav
./lac_cli decode 24.44100_lr.lac r_24.44100_lr.wav

./lac_cli decode 24.48000.lac r_24.48000.wav
./lac_cli decode 24.48000_lr.lac r_24.48000_lr.wav

./lac_cli decode 24.96000.lac r_24.96000.wav
./lac_cli decode 24.96000_lr.lac r_24.96000_lr.wav

./lac_cli decode 24.192000.lac r_24.192000.wav
./lac_cli decode 24.192000_lr.lac r_24.192000_lr.wav

echo "--------------------------------------"
echo " All encode/decode operations complete "
echo "--------------------------------------"