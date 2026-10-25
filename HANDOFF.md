# HANDOFF (lane ml-convertant)

State 2026-10-25: main 0.1.2 (CI, protection, CODEOWNERS, core crate, Q8_0). Row 4 (Q4_0) in PR from
`feat/ggml-q4_0` (worktree `C:\Projects\ml-convertant-wt-q4`), version 0.1.3 proposed; PM renumbers.
Primary checkout stays on `main`.

Rows 3-4 facts: Q8_0 and Q4_0 are bit-exact vs real llama.cpp `9d57ce456c` (compiled ggml-quants.c with MSVC from
VS 18; harness in tools/golden/). Non-finite input and subnormal-scale blocks are errors (reference UB).
Next: row 5 NF4 (against the bnb fixture in mlmf-conformance/fuel-tensor; Fuel has decode only; encoder new;
bitsandbytes is MIT). Golden harness needs gguf.h and ggml-cpu/ggml-cpu-impl.h besides the usual headers. One PR per row; PM allocates versions.
Live llama.cpp differential now due (Q4_0 done) (self-hosted, board 241).
Before the first real publish: patent sweep (ROADMAP row 9, docs/PATENT-SWEEP.md). Inputs from the mlmf
lane: PTQ calibration (min/max, percentile, entropy-histogram), INT8/INT4, dynamic/static; hits
US12099930B2 (TI histogram tail-discard, moving-average max, power-of-2 clip), US20230342613A1
(Alphaics integer-only QAT) — clear for mlmf's old code, not for us if we add those steps.
