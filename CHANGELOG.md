# Changelog

All notable changes to this workspace. One version number covers every crate (see `CLAUDE.md`).
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Pre-1.0, a breaking change raises the
second number.

## [0.1.33]

### Added
- `tests/real_checkpoints.rs` (by the mlmf lane, corpus-gated: prints SKIPPED when `C:/Models` is absent): real Qwen3-0.6B-AWQ layers (196) round-trip through `pack_hf`/`unpack_hf` and the kernel repack with bit-identical dequantized values; real SmolLM2-360M weights through `rtn_awq` and `search_awq` with SYNTHETIC activations. These cannot see a consistent column permutation: the AWQ pack order against AutoAWQ is still not verified, and no real GPTQ checkpoint or real activations were used.

## [0.1.32]

### Added
- `ml-convertant-calibrate`: `search_awq` (AWQ-style scale search: per-input-channel `s_i = mean|x_i|^alpha`, alpha on a 20-point grid, least output error on the calibration activations wins, alpha 0 is round to nearest so it is never worse), with `AwqSearch`, `Searched` (the quantized `W s`, the channel scales the caller folds into the preceding op, and weight, output and baseline reports). Output error is measured; the weight bound is proven for the effective weights.
- `docs/PATENT-SWEEP.md`: idea 18, read on file, no new search. `docs/CALIBRATION-DESIGN.md`: corrected (the channel scale cannot be folded into the per-group AWQ scale; it is returned for the caller).

## [0.1.31]

### Added
- `ml-convertant-calibrate` (new): `rtn_awq`, round-to-nearest quantization of `[OC, IC]` weights into the AWQ layout, with a weight `ErrorReport` carrying a proven per-group bound and, given a `CalibrationSet`, a measured output-error report. Bit-identical to the LoRA AWQ requantizer on shared inputs. The AWQ-style scale search is the next release. Re-exported by the facade as `calibrate`.
- `docs/PATENT-SWEEP.md`: idea 17, read on file, no new search.

## [0.1.30]

### Changed
- `docs/PATENT-SWEEP.md`: calibration re-sweep recorded as thin (two web queries W6 and W7, no patent database searched, no new row; H9 is the nearest read).

## [0.1.29]

### Added
- `ml-convertant-gptq` (new): AutoGPTQ v1 4-bit layers (`qweight`, `qzeros` stored minus one, `scales`, optional `g_idx`): `unpack_hf`, `pack_hf`, `dequantize_logical`, `to_awq_logical` and the `GptqToAwq` conversion, lossless with a zero-error report. Activation-order layers dequantize but are refused by the AWQ conversion. Re-exported by the facade as `gptq`. The Marlin repack stays in baracuda.
- `THIRD-PARTY-NOTICES.md`: AutoGPTQ (MIT) layout entry. `docs/PATENT-SWEEP.md`: idea 16, read on file, no new search.

## [0.1.28]

### Added
- `ml-convertant-lora`: `merge_lora_into_nf4` (NF4 base, same block size, f32 absmax out) and `merge_lora_into_awq` (AWQ layer in the logical layout; group-wise min/max round-to-nearest requantizer with f16 scale rounded up, not AWQ's activation-aware search). Each reports the chain's error against the exact f64 merge of the dequantized base with a proven bound. New `AwqRequantError`; the crate now depends on `ml-convertant-nf4`, `ml-convertant-awq` and `half`.
- `docs/PATENT-SWEEP.md`: idea 15, read on file, no new search.

## [0.1.27]

### Added
- `ml-convertant-mlmf`: `convert_safetensors` reads a safetensors file (dense F32/F16/BF16 tensors) and writes a GGUF with the selected tensors quantized, same `Plan`, per-tensor `ErrorReport` and outcomes as `convert_gguf`. Dimensions are reversed to GGUF `ne`, a scalar becomes a one-element vector, the caller supplies the GGUF metadata (`general.architecture` is required by MLMF's encoder), safetensors names are carried over unchanged. New `GlueError::Safetensors`. The GGUF path is refactored onto the same per-tensor helper with its tests unchanged.
- `docs/PATENT-SWEEP.md`: idea 14 (safetensors as a source), read on file, no new search.

## [0.1.26]

### Changed
- `ROADMAP.md` and `HANDOFF.md` current after the first release and both 421 rulings.

## [0.1.25]

### Added
- `ml-convertant-lora`: `LoraAdapter::with_strength` adjusts an adapter's strength after load and before merge (ruling 421 Q2): `scale` times a factor, rounded once; `0.0` leaves the base bit-identical, a negative factor subtracts the adapter. Works for `merge_lora` and `merge_lora_into_quantized`.
- `docs/PATENT-SWEEP.md`: idea 13 (strength adjustment) read against H8, no new search.

## [0.1.24]

### Added
- `ml-convertant-names`: rename LoRA adapter tensors between the PEFT and GGUF forms, both directions, for the Llama-style layout (CireSnave ruling 421, 2026-11-12); unmeasured names refused with a typed error; reports when an embedding's data must be transposed.

### Changed
- `docs/PATENT-SWEEP.md`: quotes the board 307 ruling (option C, 2026-11-01) verbatim from the board archive.

## [0.1.23]

### Changed
- `mlmf-core`, `mlmf-ggml`, `mlmf-gguf`, `mlmf-bnb`, `mlmf-safetensors` pins raised from `~0.7.x` to `~0.11.29` (crates.io latest); workspace builds and tests pass.
- `docs/PATENT-SWEEP.md`: board 307 ruled (2026-11-01, option C); per-crate re-read of the Nvidia pair US12045307B2 / US12412082B2 recorded (a reading, not legal advice).
- `HANDOFF.md` rewritten without the release hold.

## [0.1.22]

### Changed
- Docs only: HANDOFF and ROADMAP record the first-release hold (board item 307).

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
