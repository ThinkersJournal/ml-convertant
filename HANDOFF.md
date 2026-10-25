# HANDOFF (lane ml-convertant)

State 2026-10-25: main 0.1.3 (Q4_0 merged). Rebasing the queue in the PM's order: #7 NF4 (0.1.4), #8 NF4
double-quant (0.1.5), #9 AWQ (0.1.6), #10 llama.cpp differential (0.1.7), #11 LoRA (0.1.8); then #12 K-quants
Q4_K (placeholder 0.1.9). Primary checkout stays on `main`; each branch has its own `ml-convertant-wt-*`
worktree. One PR per row; the PM allocates versions at each gate.

Facts: Q8_0 and Q4_0 are bit-exact vs real llama.cpp `9d57ce456c` (compiled ggml-quants.c with MSVC from
VS 18; harnesses in tools/golden/; the build also needs gguf.h and ggml-cpu/ggml-cpu-impl.h). Non-finite
input and subnormal-scale blocks are errors (reference UB). NF4: the encoder reproduces all 8192 packed
bytes of the real bitsandbytes fixture from the original bf16 weights; Fuel has decode only.

Before the first real publish: patent sweep (ROADMAP row 9, docs/PATENT-SWEEP.md; pace queries ~1/min).
Inputs from the mlmf lane: PTQ calibration (min/max, percentile, entropy-histogram), INT8/INT4,
dynamic/static; hits US12099930B2 (TI histogram tail-discard, moving-average max, power-of-2 clip),
US20230342613A1 (Alphaics integer-only QAT): clear for mlmf's old code, not for us if we add those
steps. Claims reader: C:\Projects\tools\patent-claims.py.

Row 5b (NF4 double-quant absmax encode) is PR #8 from `feat/nf4-double-quant` (worktree `C:\Projects\ml-convertant-wt-nf4dq`), stacked on #7: dynamic map and all 256 absmax codes bit-identical to the fixture (torch linspace is FMA; python float * f32 tensor is an f32 multiply).
