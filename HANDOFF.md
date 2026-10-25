# HANDOFF (lane ml-convertant)

State 2026-10-25: main 0.1.1 (CI, branch protection, CODEOWNERS, core crate). Row 3 (Q8_0) in PR from
`feat/ggml-q8_0` (worktree `C:\Projects\ml-convertant-wt-q8`), version 0.1.2 proposed; PM renumbers.
Primary checkout stays on `main`.

Row 3 facts: Q8_0 is bit-exact vs real llama.cpp `9d57ce456c` (compiled ggml-quants.c with MSVC from
VS 18; harness in tools/golden/). Non-finite input and subnormal-scale blocks are errors (reference UB).
Next: row 4 Q4_0 the same way (new harness fn + golden vectors; `tools/golden/` needs gguf.h and
ggml-cpu/ggml-cpu-impl.h besides the files in NOTICES). One PR per row; PM allocates versions.
Live llama.cpp differential waits until after Q4_0 (self-hosted, board 241).
Before the first real publish: patent sweep (ROADMAP row 9, docs/PATENT-SWEEP.md). Inputs from the mlmf
lane: PTQ calibration (min/max, percentile, entropy-histogram), INT8/INT4, dynamic/static; hits
US12099930B2 (TI histogram tail-discard, moving-average max, power-of-2 clip), US20230342613A1
(Alphaics integer-only QAT) — clear for mlmf's old code, not for us if we add those steps.
