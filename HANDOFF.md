# HANDOFF (lane ml-convertant)

State 2026-11-12. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.23 at 7803d27):** core (`ErrorReport`, `Conversion`), ggml (Q8_0, Q4_0), kquants (Q2_K..Q6_K), nf4 (+ double
quantization), awq (unpack/pack/repack), lora (merge; merge into a quantized base for Q8_0, Q4_0, Q2_K..Q6_K), mlmf glue (GGUF
in/out, mlmf ~0.11.29), CI differential vs llama.cpp (ubuntu+windows, Q8_0..Q6_K), `docs/PATENT-SWEEP.md` (per-crate H1/H2 read).
Branch protection: 5 required checks.

**FIRST RELEASE (PM [TASK]s 2026-11-12/13; board 307 ruled C 2026-11-01; CireSnave asked the PM to publish):** the PM publishes and
allocates versions. Done: steps 1-3 (#25, merged). Open: step 4, PR #26 (`names-crate`, worktree `ml-convertant-wt-names`): MLMF-free
`ml-convertant-names` (ruling 421: Llama-style layout, both directions, others refused), versions set to 0.1.24 (proposed, the PM may
re-allocate), plus the verbatim 307 quote. Step 5: the PM publishes in order core, ggml, kquants, nf4, awq, lora, mlmf, names, facade.
Next row after #26: adapter strength/scale adjustment after load, before use (ruling 421 Q2; MLMF only loads and saves; sweep the
patents first: it touches merge arithmetic, H8).

**Unscheduled ideas (no row; start one only with a PM/CireSnave go):** safetensors glue, NF4/AWQ bases in the
LoRA-into-quantized merge, GPTQ repack, calibration (re-sweep patents first: US12099930B2 H9), IQ* K-quant formats,
differential for other formats.

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
