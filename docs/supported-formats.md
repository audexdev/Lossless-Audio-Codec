# Supported Formats

This document describes the current supported input/output domain. LAC is experimental and should not yet be treated as a long-term archival format.

## WAV Input

The current WAV reader accepts a narrow PCM domain:

| Field | Supported values |
| --- | --- |
| Container | RIFF/WAVE |
| Audio format | PCM (`1`) |
| Channels | `1` mono or `2` stereo |
| Bit depth | `16` or `24` bits per sample |
| Sample rates | `44100`, `48000`, `96000`, or `192000` Hz |
| Sample layout | interleaved little-endian PCM |

Unsupported variants should be rejected cleanly rather than decoded incorrectly. Examples include floating-point WAV, extensible WAV metadata, more than two channels, unsupported sample rates, unsupported bit depths, and malformed chunk sizes.

## Sample Ranges

Decoded samples are represented internally as signed 32-bit integers, but the supported WAV domain is narrower:

| Bit depth | Minimum | Maximum |
| ---: | ---: | ---: |
| 16 | `-32768` | `32767` |
| 24 | `-8388608` | `8388607` |

The LAC encoder validates public input samples against the configured bit depth and rejects out-of-domain values. The WAV writer currently clips samples to the target bit depth when writing restored WAV files.

Block-level codec headers under `src/codec/block/` are internal implementation surfaces. They are used by tests, but they are not a stable public API for arbitrary `int32_t` sample domains.

## LAC Container

The current `.lac` container supports:

- format version `2`
- mono or stereo streams
- LR, mid/side, or per-block stereo mode
- block sizes up to `16384` samples per channel
- fixed, FIR, and LPC predictors
- adaptive Rice, zero-run, and small-residual bin residual modes

See `docs/format.md` for bitstream details.

## Current Limits

- The format has no checksum, frame CRC, block CRC, or authenticated payload length.
- Large-file handling is still bounded by classic RIFF/WAV 32-bit size fields on output.
- Streaming encode/decode is not currently exposed as a public API.
- Malformed-input validation is being improved through security hardening issues and fuzzing work.
