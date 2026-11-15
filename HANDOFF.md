# HANDOFF (lane ml-convertant)

State 2026-11-14. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.25 at 096f1d9; 0.1.24 published to crates.io and tagged v0.1.24 by the PM 2026-11-15, all 9 crates):** core (`ErrorReport`,
`Conversion`), ggml (Q8_0, Q4_0), kquants (Q2_K..Q6_K), nf4 (+ double quantization), awq (unpack/pack/repack), lora (merge; merge into
a quantized base for Q8_0, Q4_0, Q2_K..Q6_K), mlmf glue (GGUF in/out, mlmf ~0.11.29), names (LoRA adapter tensor names PEFT<->GGUF,
Llama-style layout, others refused; ruling 421 Q1), CI differential vs llama.cpp, `docs/PATENT-SWEEP.md` (per-crate H1/H2 read; board 307 ruled C
2026-11-01, no integer dot kernel here; the K-quants do H2's per-vector steps, so the "no" rests on the missing multiply limitation alone).

**In flight (PM [TASK] go-converter-rows 2026-11-15, one PR each, in order):** done: safetensors glue (#29, 0.1.27). Now: NF4/AWQ bases in the
LoRA-into-quantized merge, branch `lora-nf4-awq-bases`, worktree `ml-convertant-wt-nf4awq`, version 0.1.28 proposed (AWQ requantizer is min/max RTN, not
AWQ calibration; say so). Next: GPTQ repack; then calibration (re-sweep patents first) and IQ*. Converters only: no integer dot-product kernel (board 307 C).

**Unscheduled ideas (no row; start one only with a PM/CireSnave go):** calibration (re-sweep patents first: US12099930B2 H9), IQ* K-quant formats,
differential for other formats, safetensors out (only if a consumer asks).

**Facts worth keeping:** Q8_0..Q6_K are byte-identical to ggml's own code compiled from llama.cpp `9d57ce456c`
(harnesses `tools/golden/*_reference.c`, -DNDEBUG for the K-quants, MSVC locally, gcc in CI). NF4 encoder reproduces a real
bitsandbytes checkpoint byte for byte. AWQ repack is identical to Fuel's, NOT checked against a real AWQ checkpoint. LoRA
merge is NOT compared with PEFT-merged weights. The glue adds `general.quantization_version` = 2 only when it converted
something and the input lacked it. One version for every crate and path-dep minimum.

**Patent sweep:** first record done; re-read triggers are in `docs/PATENT-SWEEP.md`. Google Patents throttles after a few
queries (use C:\Projects\tools\patent-claims.py); USPTO image PDFs via WebFetch downloadPdf + PyMuPDF work for US documents.

**Weekly sweep:** last 2026-11-12 (issues 0, PRs 0, roadmap open 0). Next due ~2026-11-19.

**Shell traps:** Bash tool breaks on big quoted heredocs containing apostrophes (write files with the Write tool, run Python
from a file); `gh pr create` can 504: re-check `gh pr list --head` before retrying; never grep across `C:\Projects` (walks
other lanes' worktrees, times out); worktrees `ml-convertant-wt-*` beside the primary checkout, which stays on `main`.
