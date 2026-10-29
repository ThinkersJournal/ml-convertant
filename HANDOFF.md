# HANDOFF (lane ml-convertant)

State 2026-10-29. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.21 at cf7c90b):** core (`ErrorReport`, `Conversion`), ggml (Q8_0, Q4_0), kquants (Q2_K..Q6_K), nf4 (+ double
quantization), awq (unpack/pack/repack), lora (merge; merge into a quantized base for Q8_0, Q4_0, Q2_K..Q6_K), mlmf glue (GGUF
in/out), CI differential vs llama.cpp (ubuntu+windows, Q8_0..Q6_K), `docs/PATENT-SWEEP.md` (22 patents read; 20 "outside as read", H1/H2 not read for this code).
Branch protection: 5 required checks. Every ROADMAP row is done; no PR of mine is open except the docs PR for 0.1.22.

**FIRST RELEASE HELD (PM 2026-10-29):** publish waits on CireSnave's ruling on board item 307: PATENT-SWEEP rows H1/H2
(Nvidia US12045307B2 / US12412082B2, per-vector scale plus coarse scale = the K-quant shape) are "not cleared, not read for
this code". Release prep is merged (#23: README hello, `examples/quantize_gguf.rs`, sweep wording). The PM wakes this lane
when 307 is ruled or a row appears; do nothing until then.

**Unscheduled ideas (no row; start one only with a PM/CireSnave go):** safetensors glue, NF4/AWQ bases in the
LoRA-into-quantized merge, GPTQ repack, calibration (re-sweep patents first: US12099930B2 H9), IQ* K-quant formats,
differential for other formats.

**Facts worth keeping:** Q8_0..Q6_K are byte-identical to ggml's own code compiled from llama.cpp `9d57ce456c`
(harnesses `tools/golden/*_reference.c`, -DNDEBUG for the K-quants, MSVC locally, gcc in CI). NF4 encoder reproduces a real
bitsandbytes checkpoint byte for byte. AWQ repack is identical to Fuel's, NOT checked against a real AWQ checkpoint. LoRA
merge is NOT compared with PEFT-merged weights. The glue adds `general.quantization_version` = 2 only when it converted
something and the input lacked it.

**Patent sweep:** first record done; re-read triggers are in `docs/PATENT-SWEEP.md`. Google Patents throttles after a few
queries (use C:\Projects\tools\patent-claims.py); USPTO image PDFs via WebFetch downloadPdf + PyMuPDF work for US documents.
Fuel's kernels are exposed to the Nvidia pair (US12045307B2 claims 1/14/17, US12412082B2), board 307.

**Weekly sweep:** last 2026-10-26 (issues 0/0/0; roadmap open 0). Next due ~2026-11-02.

**Shell traps:** Bash tool breaks on big quoted heredocs containing apostrophes (write files with the Write tool, run Python
from a file); `gh pr create` can 504: re-check `gh pr list --head` before retrying; worktrees `ml-convertant-wt-*` beside the
primary checkout, which stays on `main`.
