# Supported Formats

This document describes the current supported input/output domain. LAC is experimental and should not yet be treated as a long-term archival format.

## WAV Input

`lac_cli encode` accepts a narrow PCM WAV domain:

| Field | Supported values |
| --- | --- |
| Container | RIFF/WAVE |
| Audio format | PCM (`1`) |
| `fmt ` chunk | exactly one 16-byte PCM chunk before `data` |
| `data` chunk | exactly one non-empty chunk containing complete sample frames |
| Channels | `1` mono or `2` stereo |
| Bit depth | `16` or `24` bits per sample |
| Sample rates | `44100`, `48000`, `96000`, or `192000` Hz |
| Sample layout | interleaved little-endian PCM |

The RIFF size, chunk boundaries, `byte_rate`, and `block_align` fields must be consistent. Unknown ancillary chunks are skipped, including odd-sized chunks with their RIFF padding byte. Ancillary metadata is not preserved in the restored WAV.

Unsupported variants are rejected cleanly rather than decoded incorrectly. Examples include floating-point WAV, extensible WAV metadata, duplicate `fmt ` or `data` chunks, more than two channels, unsupported sample rates, unsupported bit depths, empty PCM payloads, and malformed chunk sizes.

## Sample Ranges

Decoded samples are represented internally as signed 32-bit integers, but the supported WAV domain is narrower:

| Bit depth | Minimum | Maximum |
| ---: | ---: | ---: |
| 16 | `-32768` | `32767` |
| 24 | `-8388608` | `8388607` |

The LAC encoder and WAV writer validate samples against the configured bit depth and reject out-of-domain values. Restored output is never silently clipped.

Block-level codec headers under `src/codec/block/` are internal implementation surfaces. They are used by tests, but they are not a stable public API for arbitrary `int32_t` sample domains.

## CLI Roundtrip Contract

The alpha contract is CLI-first:

```text
supported WAV -> lac_cli encode -> .lac -> lac_cli decode -> restored WAV
```

For supported input, the restored WAV has the same PCM samples, channel count, sample rate, and bit depth. The restored file is a canonical PCM WAV with a 16-byte `fmt ` chunk followed by one `data` chunk. When the PCM payload size is odd, the writer appends the required zero RIFF padding byte after the `data` payload. Ancillary chunks and their metadata are intentionally not copied. Encode defaults to per-block stereo selection for stereo input; `--stereo-mode=lr` and `--stereo-mode=ms` force a mode. Mono input always uses LR mode metadata.

Encode and decode reject input and output paths that refer to the same file, including a second check immediately before output publication. The CLI writes a complete output into a temporary sibling directory and replaces the final path only after the writer closes successfully. A failed write or publication leaves an existing final output untouched, and replacing an existing symlink or hardlink output does not stream bytes into its prior target. Publication is not `fsync`-backed crash durability, and the CLI is not a filesystem access-control boundary for directories concurrently modified by untrusted processes.

The alpha CLI surface is:

| Command or option | Behavior |
| --- | --- |
| `lac_cli encode input.wav output.lac` | encode a supported WAV; stereo input defaults to per-block stereo selection |
| `lac_cli decode input.lac output.wav` | decode one supported canonical version `2` or `3` `.lac` stream to canonical PCM WAV |
| `--stereo-mode=lr` | force LR stereo payloads during encode |
| `--stereo-mode=ms` | force mid/side stereo payloads during encode |
| `--threads=N` | cap encode or decode workers to positive integer `N`; overrides `LAC_THREADS` |
| `LAC_THREADS=N` | cap encode or decode workers when `--threads=N` is absent |
| `--no-partitioning` | disable residual partitioning during encode |

The CLI may overwrite an existing output file when it is distinct from the input file. Flags beginning with `--debug-` are diagnostic implementation aids and are not part of the stable alpha contract.

`lac_cli selftest` is a built-in diagnostic command. It is useful for smoke testing but is not part of the stable alpha encode/decode contract.

## LAC Container

The current `.lac` container supports:

- canonical encode format version `3`; legacy version `2` remains decode-compatible
- mono or stereo streams
- LR, mid/side, or per-block stereo mode
- block sizes up to `16384` samples per channel
- fixed, FIR, and LPC predictors
- adaptive Rice, zero-run, small-residual bin, and static Rice residual modes

See `docs/format.md` for bitstream details.

## Current Limits

- The format has no checksum, frame CRC, block CRC, or authenticated payload length. Version `3` block lengths are structural boundaries only.
- The WAV reader and LAC decoder reject decoded PCM output above 1 GiB.
- `lac_cli decode` rejects compressed `.lac` input files above 1 GiB before loading them into memory.
- The decoder rejects non-canonical structural metadata, trailing payload bytes, out-of-range restored samples, and output that cannot fit classic RIFF/WAV 32-bit size fields.
- The decoder rejects more than `1048576` blocks and non-final blocks shorter than `256` samples to bound block-table metadata and tiny-block decode work.
- Large-file handling is still bounded by classic RIFF/WAV 32-bit size fields on output.
- On Windows, CLI paths currently pass through narrow argument and stream APIs. Non-ASCII paths outside the active code page are not guaranteed.
- Streaming encode/decode is not currently exposed as a public API.
- Structurally valid payload corruption can still produce different in-range PCM because the format has no integrity field.
- Malformed-input validation continues to be improved through security hardening issues and fuzzing work.
