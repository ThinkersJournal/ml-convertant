# Changelog

All notable changes to this workspace. One version number covers every crate (see `CLAUDE.md`).
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Pre-1.0, a breaking change raises the
second number.

## [0.1.21]

### Added
- `examples/quantize_gguf.rs` and a README hello (GGUF in, Q8_0 or Q4_0 out, error per tensor).

### Changed
- `docs/PATENT-SWEEP.md`: verdicts read "outside as read" (a reading, not a clearance); the Nvidia pair US12045307B2 and US12412082B2 are stated as not read for this code.

## [0.1.20]

### Changed
- Docs only: HANDOFF and ROADMAP brought current (rows 8 and 11 done).

## [0.1.19]

### Added
- `ml-convertant-lora`: merge into a quantized base (`merge_lora_into_quantized`, `QuantFormat`, `RequantError`) for Q8_0, Q4_0 and
  Q2_K to Q6_K: dequantize, merge in f32, requantize; error of the chain against an f64 reference of the exact merge of the
  dequantized base, with a proven bound (merge bound + the format's requantization bound).

## [0.1.18]

### Changed
- `HANDOFF.md` rewritten for the throttle boundary.

## [0.1.17]

### Added
- `ml-convertant-mlmf`: GGUF in, GGUF out (`convert_gguf`, `Plan`, `Target`): reads with `mlmf-gguf`, converts the selected
  dense F32/F16/BF16 tensors to Q8_0, Q4_0 or Q2_K to Q6_K with a per-tensor error report, and writes with MLMF's encoder.
  Metadata and unselected tensors are carried over unchanged; a file MLMF cannot fully describe is refused;
  `general.quantization_version` (2) is added, and reported, when something was converted and the input lacked it.

## [0.1.15]

### Changed
- `ROADMAP.md` statuses brought current (rows 0, 6, 7, 7b, 10-10d done; rows 8 and 11 open with an owner).

## [0.1.14]

### Added
- CI differential extended to the K-quants Q2_K to Q6_K: the pinned llama.cpp reference is built with `-DNDEBUG` and
  compared byte for byte on seeded random input; `ml-convertant-kquants` example `convert_kquants_file`.

## [0.1.13]

### Added
- `ml-convertant-kquants`: Q2_K and Q3_K (`quantize_q2_k`, `quantize_q3_k` and their dequantizers, bounds and `Q2K`, `Q3K`),
  bit-exact against ggml's reference on 60 super-blocks each. All of ggml's K-quants Q2 to Q6 are now covered.

## [0.1.12]

### Added
- `ml-convertant-kquants`: Q6_K (`quantize_q6_k`, `dequantize_q6_k`, `q6_k_bound`, `Q6K`), bit-exact against ggml's reference
  on 60 super-blocks.

## [0.1.11]

### Added
- `ml-convertant-kquants`: Q5_K (`quantize_q5_k`, `dequantize_q5_k`, `q5_k_bound`, `Q5K`), bit-exact against ggml's reference
  on 60 super-blocks; the Q4_K/Q5_K scale search is shared.

## [0.1.10]

### Added
- `ml-convertant-kquants`: Q4_K (`quantize_q4_k`, `dequantize_q4_k`, `q4_k_bound`, `Q4K`), bit-exact against ggml's
  reference at llama.cpp `9d57ce456c` on 60 super-blocks; per-input error bound.

## [0.1.9]

### Added
- `ml-convertant-lora`: offline LoRA merge (`merge_lora`, `LoraAdapter`, `LoraMerge`, `lora_merge_bound`) in f32 with a
  fixed accumulation order, reported against an f64 reference with a proven bound. Not compared against PEFT-merged weights.

## [0.1.8]

### Added
- `docs/PATENT-SWEEP.md`: first patent sweep record (22 patents read, all clear for conversions; re-read triggers listed).

## [0.1.7]

### Added
- `.github/workflows/differential.yml`: on hosted ubuntu and windows runners, builds ggml's reference from the
  pinned llama.cpp commit (`9d57ce456c`) and compares Q8_0 and Q4_0 byte for byte on seeded random input;
  `ml-convertant-ggml` example `convert_file`; `tools/golden/gen_random_f32.py`.

## [0.1.6]

### Added
- `ml-convertant-awq`: AWQ (GEMM) unpack, pack and OC-major kernel repack (`unpack_hf`, `pack_hf`,
  `repack_for_kernel`, `HfToKernel`); lossless, reported as zero error; word-identical to Fuel's repack on
  seeded layers. Not checked against a real AWQ checkpoint's bytes.

## [0.1.5]

### Added
- `ml-convertant-nf4`: double quantization of the per-block absmax (`dynamic_map_8bit_signed`,
  `quantize_absmax`, `quantize_absmax_with_offset`, `DoubleQuant`); the dynamic map and the 8-bit codes
  are bit-identical to a real bitsandbytes checkpoint.

## [0.1.4]

### Added
- `ml-convertant-nf4`: bitsandbytes NF4 encode/decode (`quantize_nf4`, `dequantize_nf4`,
  `dequantize_nested_absmax`, `Nf4`), byte-exact against a real bitsandbytes checkpoint slice; proven
  error bound; fixture and provenance under `tests/fixtures/bnb_double_quant/`.

## [0.1.3]

### Added
- `ml-convertant-ggml`: Q4_0 (`quantize_q4_0`, `dequantize_q4_0`, `Q4_0`), bit-exact against llama.cpp
  `9d57ce456c`; shared `BlockError` (`Q8_0Error`/`Q4_0Error` are aliases).

## [0.1.2]

### Added
- `ml-convertant-ggml`: Q8_0 (`quantize_q8_0`, `dequantize_q8_0`, `Q8_0`), bit-exact against llama.cpp
  `9d57ce456c`; proven error bound.

## [0.1.1]

### Added
- `.github/CODEOWNERS`.

### Changed
- `cargo-deny-action` pinned to its commit SHA rather than the annotated tag object.

## [0.1.0]

### Added
- `ml-convertant-core`: `ErrorReport` (the measured-error contract), `Conversion`, `Converted`.
- Workspace and the `ml-convertant` facade crate.

## [0.0.1]

### Added
- CI on hosted runners (fmt, clippy, tests, doc, cargo-deny, cargo-audit), branch protection,
  `docs/DESIGN.md`, repository agreements.

## [0.0.0]

- Name reservation.
