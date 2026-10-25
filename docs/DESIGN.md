# ML-ConvertAnt design (one page)

2026-10-25. Scope per `CLAUDE.md`. Inputs measured at: mlmf `34afd0e^` (removed quantization code),
fuel `fuel-quantized` (`awq_repack.rs`, `k_quants.rs`) and `fuel-cuda-backend/src/baracuda/quant_w4a16.rs`,
mlmf `mlmf-ggml/tests/ggml-types.tsv` (llama.cpp `9d57ce456c`), mlmf-bnb fixture.

## 1. Crate layout (one workspace, one version number)

| crate | holds |
|---|---|
| `ml-convertant-core` | `ErrorReport`, the `Conversion` trait, block/shape helpers. No I/O, no mlmf. |
| `ml-convertant-ggml` | Q8_0, Q4_0 (later K-quants) block encode/decode, ggml layout. |
| `ml-convertant-nf4` | bitsandbytes NF4 (+ double-quant) encode/decode. |
| `ml-convertant-awq` | AWQ unpack/repack (HF GEMM layout <-> kernel layouts). |
| `ml-convertant-lora` | LoRA merge into base weights. |
| `ml-convertant` | Facade re-exporting the above; later an `mlmf` feature for read -> convert -> write. |

All crates share one version (portfolio rule). Crates are added as their row starts; the 0.0.0
placeholder name `ml-convertant` stays the facade. Pure safe Rust, scalar reference first; SIMD only
after the scalar path is bit-exact and only if behind the same tests.

## 2. Keystone first slice

f32 <-> **Q8_0** (32 elements/block: f16 scale `d` + 32 `i8`, 34 bytes; geometry from
`ggml-types.tsv`). Encode `d = amax/127`, `q = round_half_away(x/d)`, scale stored as f16 as ggml does.
Delivered with the §3 report, a proven error bound, and a differential against llama.cpp's
`quantize_row_q8_0_ref`. Then Q4_0, NF4, AWQ repack, LoRA merge (order in `ROADMAP.md`).

## 3. Error-measurement contract

Every lossy conversion returns `(output, ErrorReport)`; a conversion that cannot report does not exist.
`ErrorReport`, measured **against the source values in f64**, element-wise over the whole tensor:

- `n` elements compared; `max_abs` = max |x - x'| and `max_abs_index`; `mean_abs` = mean |x - x'|;
  `rmse` = sqrt(mean (x - x')^2). Relative error is derived by the caller, never silently substituted.
- `nonfinite_in` / `nonfinite_out` counts. NaN/Inf are counted, excluded from the three metrics, and
  never hidden: a round trip that changes finiteness is an error, not a statistic.
- `bound: Option<f64>` = the format's proven worst-case |x - x'| for this input (Q8_0: `d_f16/2` plus
  the scale-rounding term, per block, stated in the code doc). Tests assert `max_abs <= bound`.
- `bit_exact_vs: Option<ReferenceId>`: set only when the output bytes were compared equal to a named
  reference (e.g. `llama.cpp@9d57ce456c`). Lossless conversions (repack, unpack) must report
  `max_abs == 0` and be bit-exact; LoRA merge reports error vs. an f64 merge of the same inputs.

Where a reference exists the test is byte equality, not a tolerance.

## 4. Test strategy

Property tests for the bound; golden vectors generated once from the reference at a pinned commit and
committed with provenance (hosted CI needs no llama.cpp). A live differential against a built
llama.cpp is a self-hosted-runner job, not `#[ignore]`.

## 5. Provenance plan

Nothing is copied before its origin is verified. `THIRD-PARTY-NOTICES.md` lists, per item: origin
URL + commit, licence, files. Known sources and their treatment:

- **Q8_0/Q4_0 geometry and golden vectors:** ggml/llama.cpp (MIT) at `9d57ce456c`; we write our own
  implementation of the published layout and credit the vectors.
- **mlmf `34afd0e^` quantization code:** our own (MIT OR Apache-2.0), but its calibration design is
  re-derived, not bulk-restored: it never measured error against a reference.
- **Fuel `fuel-quantized` (k_quants, awq_repack):** ours, but `k_quants` descends from candle
  (MIT/Apache) and the IQ/TQ parts from ggml; verify against Fuel's own THIRD-PARTY-NOTICES before any
  copy, and carry their credit. `awq_repack` is Fuel-original; confirm with the Fuel lane.
- **NF4:** bitsandbytes (MIT) codebook, verified against the Qwen3-0.6B bnb fixture (Apache-2.0,
  with its licence text) already vendored by mlmf-conformance and fuel-tensor.

## 6. Open questions for the PM

1. Fuel `fuel-quantized` owns k_quants today; do we later absorb it (Fuel depends on us) or stay a
   sibling? Not blocking the keystone.
2. Self-hosted runner for a live llama.cpp differential: want it now or after Q4_0?
