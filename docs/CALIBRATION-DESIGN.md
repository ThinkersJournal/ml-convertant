# Calibration crate design note (ROADMAP row 17, proposed)

Status: design only, no code. Lane ml-convertant, 2026-11-15. Patent reading: thin (see `PATENT-SWEEP.md`, W6/W7: no patent
database searched, H9 US12099930B2 nearest). Not legal advice.

## Name and scope

Crate `ml-convertant-calibrate` (name free on crates.io at this date; PM claims it at 0.0.0). Converter only: it chooses
quantization parameters from sample activations and then encodes weights with them. No integer dot-product kernel, no
inference (board 307 C). Nothing here touches the model's file; the mlmf glue stays the file layer.

## Inputs

- `W`: dense weights, `[OC, IC]` row-major (the layout the lora and awq crates already use).
- `X`: calibration activations for that layer, `[N, IC]` row-major, f32, supplied by the caller. We never run the model,
  so the caller (Fuel) owns how `X` is collected. We take only per-input-channel statistics and, for the error report, the
  samples themselves.
- A target: AWQ 4-bit (group size `gs`), later NF4 and the ggml types.

## Step 1: RTN baseline (ships first)

Round-to-nearest with min/max per group, asymmetric 4-bit, as `lora::awq_base` already does. Returns a `LogicalAwq` plus an
`ErrorReport`. Reported errors are on two axes: weight error (`||W - W'||`) and, when `X` is given, output error
(`||XW^T - XW'^T||` per element). The weight-error bound is the proven half-step bound per group; the output error is
measured only.

## Step 2: AWQ-style scale search

Per input channel `j`, `s_j = mean|X_:,j|^alpha`, `alpha` on a grid of 20 values in `[0, 1]` (the published method's
grid). For each `alpha`: scale `W_:,j` by `s_j`, quantize with Step 1, fold `1/s_j` back, measure output error on `X`; keep
the `alpha` with the least error. Output is a `LogicalAwq` whose scales already include the fold-back, so no activation
rescale is required at inference (we do not emit a "scale the previous layer" instruction; that is kernel-side).
Independent re-derivation from the AWQ paper (arXiv 2306.00978, MIT-licensed code not copied); the in-tree AWQ format
stays AutoAWQ-compatible. Deterministic: grid order fixed, ties go to the smaller `alpha`.

## API sketch

```rust
pub struct CalibrationSet { ic: usize, n: usize, x: Vec<f32> }       // validated shape, finite values
pub struct AwqCalibration { group_size: usize, grid: usize }          // grid default 20
pub struct Calibrated { pub awq: LogicalAwq, pub alpha: f32, pub report: ErrorReport, pub baseline: ErrorReport }
pub fn rtn_awq(w, oc, ic, group_size, Option<&CalibrationSet>) -> Result<Calibrated, CalibrateError>;
pub fn search_awq(w, oc, ic, &CalibrationSet, &AwqCalibration) -> Result<Calibrated, CalibrateError>;
```

`baseline` is the RTN report on the same `X`, so every result states whether the search helped. A search that does not beat
RTN on output error returns the RTN result (alpha 0) rather than a worse one. A `Conversion` impl wraps both for the
`ErrorReport` contract.

## What we deliberately do not do (patent and scope)

- No histogram, no tail discard, no moving-average maximum, no power-of-two clipping (H9). Scales come from a mean of
  absolute activations and a grid search on output error.
- No training, no gradient step (H8, H10 to H12, H14). Search is forward-only and grid-based.
- No format auto-selection from measured error (H4, H18): the caller picks the target.
- GPTQ's Hessian solve is out of this row; it needs its own sweep and its own go.

## Verification plan

- RTN: bit-identical to `lora::awq_base`'s requantizer on shared inputs (existing test oracle).
- Search: property tests (never worse than RTN on `X`; deterministic; alpha 0 equals RTN) and a hand-worked tiny case where
  one outlier channel makes `alpha > 0` strictly better.
- Mutation checks as in other crates (flip the grid order, drop the fold-back, skip the RTN fallback) each caught.
- Not verified: no comparison against AutoAWQ's own scales on a real model; the README will say so.

## Open for the PM

1. Name `ml-convertant-calibrate`: ok? 2. Ship RTN first (one PR) then the search (second PR)? Recommended yes.
3. Output error needs `X`; callers without activations get RTN weight error only. Recommended: accept.
