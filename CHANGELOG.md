# Changelog

All notable changes to this workspace. One version number covers every crate (see `CLAUDE.md`).
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Pre-1.0, a breaking change raises the
second number.

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
