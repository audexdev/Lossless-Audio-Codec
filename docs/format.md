# LAC Bitstream Format

This document describes the current experimental `.lac` format implemented by this repository. It is a developer reference, not a frozen external standard.

## Conventions

- Multi-bit fields are written most-significant bit first by `BitWriter`.
- Byte-sized fields are written as 8 bits and are byte-aligned when noted.
- Signed residuals are generally mapped with zigzag coding:
  - `0 -> 0`
  - `-1 -> 1`
  - `1 -> 2`
  - `-2 -> 3`
- A decoder should treat malformed metadata as a decode error, even where the current implementation is still being hardened.

## Top-Level Layout

```text
FrameHeader
u32 block_count
u32 block_size[block_count]
BlockPayload[block_count]
```

The block-size table gives the number of samples per channel in each block. Every channel payload in the same block uses the same `block_size`.

## Frame Header

The frame header is 80 bits, currently 10 bytes:

| Field | Bits | Meaning |
| --- | ---: | --- |
| sync | 16 | `0x4C41` (`LA`) |
| version | 8 | current format version, `2` |
| channels | 8 | `1` mono or `2` stereo |
| stereo_mode | 8 | `0` LR, `1` mid/side, `2` per-block stereo |
| sample_rate_low | 16 | low 16 bits of sample rate |
| sample_rate_high | 8 | high 8 bits of sample rate |
| bit_depth | 8 | `16` or `24` |
| reserved | 8 | written as `0` |

Supported sample rates are currently:

- 44100 Hz
- 48000 Hz
- 96000 Hz
- 192000 Hz

For mono streams, `stereo_mode` is written as `0`.

## Stereo Modes

### LR

Left and right channels are encoded as independent channel blocks.

```text
ChannelBlock(left)
ChannelBlock(right)
```

### Mid/Side

The encoder stores:

```text
mid  = floor((left + right) / 2)
side = left - right
```

The decoder reconstructs left/right from the decoded mid and side vectors.

### Per-Block Stereo

When `stereo_mode == 2`, each block starts with an 8-bit stereo flag:

| Value | Meaning |
| ---: | --- |
| `0` | block uses LR payloads |
| `1` | block uses mid/side payloads |

Values other than `0` or `1` are non-canonical and should be rejected by future hardened decoders.

## Channel Block Layout

Each channel block has this structure:

```text
u8 predictor_type
u8 predictor_order
if predictor_type == LPC:
    i16_q15 coeff[1..predictor_order]
u8 residual_control
partition metadata
residual payload
byte padding to next byte boundary
```

### Predictor Types

| Value | Predictor |
| ---: | --- |
| `0` | fixed predictor |
| `1` | FIR predictor |
| `2` | LPC predictor |

Fixed predictors currently use orders 0 through 4. The FIR predictor currently uses a predefined two-tap filter. LPC coefficients are stored as signed Q15 coefficients.

## Residual Control

The residual control byte contains the partition and default residual-mode metadata.

| Bits | Meaning |
| --- | --- |
| 7 | partition flag |
| 6..5 | default residual mode |
| 4 | reserved |
| 3..0 | partition order |

If the partition flag is unset, the block has one partition. If set, partition count is `1 << partition_order`.

Current limits:

- minimum partition size: 32 samples
- maximum partition order: 8

Partition sizes are computed as:

```text
base = block_size >> partition_order
all partitions except the final one use base
final partition uses block_size - base * (partition_count - 1)
```

Each partition then stores 7 bits of metadata:

```text
u2 residual_mode
u5 initial_k
```

## Residual Modes

### Mode 0: Adaptive Rice

Each residual is encoded with signed zigzag mapping and Rice coding using the current `k`. After every residual, `k` is adapted from the accumulated unsigned residual history.

### Mode 1: Zero-Run

Zero-run mode uses 2-bit token tags:

| Tag | Meaning |
| ---: | --- |
| `00` | normal Rice-coded residual |
| `01` | zero run |
| `10` | 32-bit zigzag escape residual |
| `11` | reserved / invalid |

Zero-run lengths are encoded as unsigned Rice values with `k = 2`, then offset by `ZERO_RUN_MIN_LENGTH` (`4`).

### Mode 2: Small-Residual Bin Mode

Bin mode uses compact tags for common residuals:

| Tag | Meaning |
| ---: | --- |
| `00` | residual `0` |
| `01` | residual `+1` or `-1`, followed by a sign bit |
| `10` | residual `+2` or `-2`, followed by a sign bit |
| `11` | fallback Rice-coded residual |

The fallback path uses the same adaptive `k` model.

## Padding

Each channel block is flushed to the next byte boundary after residual encoding. Padding bits do not carry data.

## Integrity

The current format does not include a checksum, frame CRC, block CRC, or authenticated length field. Decoders should therefore validate all structural fields strictly and reject trailing garbage, impossible block sizes, invalid residual tags, and non-canonical metadata.

## Compatibility

The format version is currently `2`, but the format is still experimental. Future work may add stronger validation, checksums, fuzzed compatibility tests, streaming decode constraints, or a frozen public specification.
