# HANDOFF (lane ml-convertant)

State 2026-10-26: main 0.1.5 (Q8_0, Q4_0, NF4 + double quant merged). Rebasing the queue in the PM's order:
#9 AWQ (0.1.6, this), #10 llama.cpp differential (0.1.7), #11 LoRA (0.1.8); then the K-quants stack
#12 Q4_K, #13 Q5_K, #14 Q6_K, #15 Q2_K+Q3_K (0.1.9 to 0.1.12 by merge order, each rebased after its parent).
Primary checkout stays on `main`; each branch has its own `ml-convertant-wt-*` worktree. One PR per row; the PM
allocates versions at each gate.

Facts: Q8_0, Q4_0 and the K-quants are bit-exact vs real llama.cpp `9d57ce456c` (ggml-quants.c compiled with MSVC
from VS 18; harnesses in tools/golden/; K-quants need -DNDEBUG; the build also needs gguf.h and
ggml-cpu/ggml-cpu-impl.h). Non-finite input and subnormal-scale blocks are errors (reference UB). NF4: the encoder
reproduces all 8192 packed bytes of the real bitsandbytes fixture from the original bf16 weights; the dynamic map
and absmax codes are bit-identical too. AWQ: identical to Fuel's repack on 5 seeded cases and consistent with
AutoAWQ's layout; NOT checked against a real AWQ checkpoint's bytes.

Patent sweep (ROADMAP row 9): INCOMPLETE. Draft on branch `docs/patent-sweep` (`docs/PATENT-SWEEP.DRAFT.md`, no PR):
three US hits read and clear (Nvidia US12045307B2 and US12412082B2, Intel US20250117639A1); Google Patents is
throttling this IP (503); a retry loop writes `sweep_out2.txt` in the session scratchpad. The USPTO image-PDF route
(WebFetch downloadPdf + PyMuPDF) works for US documents. Unread: TI US12099930B2, Alphaics US20230342613A1 (only if
calibration is built), EP4341862A1, queries Q3-Q16. Rename the draft to PATENT-SWEEP.md only when complete.
Fuel's kernels are exposed to the Nvidia pair (board 307).

TODO after #10 lands: add q4_k..q6_k to the CI differential (build tools/golden/q*_k_reference.c with -DNDEBUG,
convert_file support). Candidates: LoRA merge into a quantized base (ROADMAP row 11), GPTQ repack, calibration.
