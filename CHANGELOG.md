# Changelog

All notable changes to this workspace. One version number covers every crate (see `CLAUDE.md`).
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Pre-1.0, a breaking change raises the
second number.

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
