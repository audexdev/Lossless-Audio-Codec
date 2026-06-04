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
repeat block_count:
    u32 block_size
    u32 compressed_size_bytes
BlockPayload[block_count]
```

The version `3` block table gives the number of samples per channel and the complete encoded byte length for each block. `compressed_size_bytes` covers the optional per-block stereo flag and every byte-padded channel block. Every channel payload in the same block uses the same `block_size`.

Current top-level limits:

- `block_count` must be non-zero.
- `block_count` must not exceed `1048576`, and the complete block table must be present before allocation.
- Each `block_size` must be non-zero and no larger than `16384` samples per channel.
- Each `compressed_size_bytes` must be non-zero, and their sum must exactly match the remaining frame payload bytes.
- Every non-final block must contain at least `256` samples per channel. The final block may be shorter.
- The total declared sample count must fit within the implementation's 1 GiB decoded-PCM allocation limit and the classic RIFF/WAV output size limit.

Version `2` streams use the legacy table layout:

```text
FrameHeader(version = 2)
u32 block_count
u32 block_size[block_count]
BlockPayload[block_count]
```

Version `2` remains decode-compatible, but it does not carry encoded block boundaries and is decoded serially.

## Frame Header

The frame header is 80 bits, currently 10 bytes:

| Field | Bits | Meaning |
| --- | ---: | --- |
| sync | 16 | `0x4C41` (`LA`) |
| version | 8 | current encoder format version, `3`; legacy decode also accepts `2` |
| channels | 8 | `1` mono or `2` stereo |
| stereo_mode | 8 | `0` LR, `1` mid/side, `2` per-block stereo |
| sample_rate_low | 16 | low 16 bits of sample rate |
| sample_rate_high | 8 | high 8 bits of sample rate |
| bit_depth | 8 | `16` or `24` |
| reserved | 8 | must be `0` |

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

```text
left  = mid + ((side + (side & 1)) >> 1)
right = left - side
```

The `side & 1` term is part of the format behavior. It preserves the rounding needed when `side` is odd.

### Per-Block Stereo

When `stereo_mode == 2`, each block starts with an 8-bit stereo flag:

| Value | Meaning |
| ---: | --- |
| `0` | block uses LR payloads |
| `1` | block uses mid/side payloads |

Values other than `0` or `1` are non-canonical and are rejected.

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

Fixed predictors use orders 0 through 4. The FIR predictor uses exactly order `2` with a predefined two-tap filter. LPC predictors use orders 1 through 32, and their order must be smaller than the channel block size. LPC coefficients are stored as signed Q15 coefficients.

### Predictor Reconstruction

For fixed predictors, the first `order` samples are stored as raw residual values:

```text
sample[i] = residual[i] for i < order
```

For later samples, reconstruction is:

```text
sample[i] = residual[i] + prediction[i]
```

Fixed predictor formulas:

| Order | `prediction[i]` |
| ---: | --- |
| `0` | `0` |
| `1` | `sample[i - 1]` |
| `2` | `2 * sample[i - 1] - sample[i - 2]` |
| `3` | `3 * sample[i - 1] - 3 * sample[i - 2] + sample[i - 3]` |
| `4` | `4 * sample[i - 1] - 6 * sample[i - 2] + 4 * sample[i - 3] - sample[i - 4]` |

The FIR predictor currently has order `2`, taps `{3, -1}`, and shift `2`. The first two samples are raw residual values. Later samples use:

```text
prediction[i] = (3 * sample[i - 1] - sample[i - 2]) >> 2
sample[i] = residual[i] + prediction[i]
```

For LPC, coefficients are stored as signed 16-bit Q15 values. Every sample uses the available preceding reconstructed samples, up to `order` taps. For early samples, taps that would refer before the start of the block are omitted:

```text
available_taps = min(i, order)
prediction[i] = sum(coeff[j] * sample[i - j] for j = 1..available_taps) >> 15
sample[i] = residual[i] + prediction[i]
```

For `i == 0`, `available_taps` is zero and the prediction is zero.

## Residual Control

The residual control byte contains the partition and default residual-mode metadata.

| Bits | Meaning |
| --- | --- |
| 7 | partition flag |
| 6..5 | default residual mode |
| 4 | reserved, must be `0` |
| 3..0 | partition order |

If the partition flag is unset, the block has one partition. If set, partition count is `1 << partition_order`.

Current limits:

- minimum partition size: 32 samples
- maximum partition order: 8

The partition flag must be set when `partition_order > 0` and unset when `partition_order == 0`. The default residual mode must be `0`, `1`, or `2`, and it must match the first partition metadata entry. Partitioned blocks use stateless Rice adaptation inside each partition. Unpartitioned blocks use stateful Rice adaptation.

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

The metadata entry is present even when the block is unpartitioned, in which case the partition count is one.

## Residual Modes

### Mode 0: Adaptive Rice

Each residual is encoded with signed zigzag mapping and Rice coding using the current `k`. After every residual, `k` is adapted from the accumulated unsigned residual history.

#### Zigzag Mapping

Signed residuals are mapped to unsigned values as:

```text
sign_mask = UINT32_MAX when residual < 0, otherwise 0
u = (uint32(residual) << 1) ^ sign_mask
```

Decoding maps back with:

```text
residual = (u >> 1)              when (u & 1) == 0
residual = -((u >> 1) + 1)       when (u & 1) == 1
```

#### Rice Coding

For unsigned value `u` and parameter `k`:

```text
q = u >> k
r = u & ((1 << k) - 1)
write q one bits
write one zero bit
write r as k bits when k > 0
```

The initial `k` for each residual segment comes from that partition's `u5 initial_k` metadata. `k` is updated after each logical residual sample, including samples represented by zero-run tokens.

#### Stateless Adaptation

Partitioned blocks use stateless adaptation. For each partition, initialize:

```text
sum = 0
count = 0
current_k = initial_k
```

After each residual sample with unsigned value `u`:

```text
sum += u
count += 1
mean = (sum + (count >> 1)) / count
k = 0
while ((1 << k) < mean && k < 31):
    k += 1
current_k = k
```

#### Stateful Adaptation

Unpartitioned blocks use stateful adaptation. For the single residual segment, initialize:

```text
sum = 0
count = 0
current_k = initial_k
previous_sum = 0
window_index = 0
window_filled = 0
window_sum = 0
recent_u[256] = all zero
large_flags[96] = all zero
zero_flags[96] = all zero
large_q_count = 0
zero_q_count = 0
```

After each residual sample, update `sum` and `count`, then compute the next `current_k`:

```text
current_u = sum - previous_sum
previous_sum = sum

micro_index = (count - 1) % 96
large_q_count -= large_flags[micro_index]
zero_q_count -= zero_flags[micro_index]

if window_filled < 256:
    window_filled += 1
else:
    window_sum -= recent_u[window_index]

recent_u[window_index] = current_u
window_sum += current_u

mean = (sum + (count >> 1)) / count
k = 0
while ((1 << k) < mean && k < 31):
    k += 1

q_base = 0 if k >= 31 else (current_u >> k)
is_large = 1 if q_base > 3 else 0
is_zero = 1 if q_base == 0 else 0

large_q_count += is_large
zero_q_count += is_zero
large_flags[micro_index] = is_large
zero_flags[micro_index] = is_zero

bias = 0
if window_filled > 0 and mean > 0:
    local_mean = (window_sum + (window_filled >> 1)) / window_filled
    if local_mean * 3 > mean * 4:
        bias = 1
    else if local_mean * 4 + 3 < mean * 3:
        bias = -1

if window_index + 1 >= 96 or window_filled >= 96:
    window_size = min(window_filled, 96)
    if large_q_count * 4 >= window_size * 3:
        bias = min(bias + 1, 1)
    else if zero_q_count * 5 >= window_size * 4:
        bias = max(bias - 1, -1)

current_k = clamp(k + bias, 0, 31)
window_index = (window_index + 1) % 256
```

### Mode 1: Zero-Run

Zero-run mode uses 2-bit token tags:

| Tag | Meaning |
| ---: | --- |
| `00` | normal Rice-coded residual |
| `01` | zero run |
| `10` | 32-bit zigzag escape residual |
| `11` | reserved / invalid |

Zero-run lengths are encoded as unsigned Rice values with `k = 2`, then offset by `ZERO_RUN_MIN_LENGTH` (`4`).

For tag `00`, exactly one residual is decoded with the current Rice `k`, then the adaptive model is updated with that residual's unsigned value.

For tag `01`, the encoded run value is decoded with unsigned Rice `k = 2`, then:

```text
run_length = encoded_value + 4
```

The decoder emits `run_length` zero residuals. The adaptive model is updated once per emitted zero residual; `sum` does not change and `count` increases by one for each zero.

For tag `10`, a 32-bit zigzag value follows and is decoded as one residual sample. The adaptive model is updated with that residual's unsigned value.

### Mode 2: Small-Residual Bin Mode

Bin mode uses compact tags for common residuals:

| Tag | Meaning |
| ---: | --- |
| `00` | residual `0` |
| `01` | residual `+1` or `-1`, followed by a sign bit |
| `10` | residual `+2` or `-2`, followed by a sign bit |
| `11` | fallback Rice-coded residual |

The fallback path uses the same adaptive `k` model.

Every bin token represents exactly one residual sample. Tags `00`, `01`, and `10` update the adaptive model with unsigned values `0`, `2 or 1`, and `4 or 3` respectively after sign reconstruction. Tag `11` decodes one Rice-coded residual using the current `k`, then updates the same adaptive model.

## Padding

Each channel block is flushed to the next byte boundary after residual encoding. Padding bits must be zero. Non-zero padding is rejected as non-canonical.

## Integrity

The current format does not include a checksum, frame CRC, block CRC, or authenticated length field. Version `3` compressed block lengths are structural boundaries, not integrity protection. Decoders validate structural fields strictly and reject trailing garbage, impossible block sizes, mismatched version `3` payload lengths, invalid residual tags, Rice values or predictor reconstruction outside the signed 32-bit domain, non-zero reserved fields or padding, and non-canonical metadata. Without an integrity field, a modified payload can still decode successfully if it remains structurally valid and produces in-range PCM samples.

## Compatibility

The encoder currently emits format version `3`, but the format is still experimental. Version `3` adds compressed block byte lengths so blocks can be validated and decoded independently. Future work may add stronger validation, checksums, fuzzed compatibility tests, streaming decode constraints, or a frozen public specification.

The decoder retains serial compatibility for canonical version `2` streams. Hardened decoders may reject version `2` byte sequences that older permissive decoders accepted when those sequences contain non-canonical reserved fields, metadata, padding, stereo flags, block tables, or trailing payload bytes.
