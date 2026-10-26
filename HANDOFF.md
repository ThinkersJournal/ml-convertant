# HANDOFF (lane ml-convertant)

State 2026-10-26 (THROTTLED until ~2026-10-28T01:00Z per C:\Projects\THROTTLE.md: gates, fixes, security only; no
new rows; no avoidable builds; box-memory rule CARGO_BUILD_JOBS=2, one cargo at a time).

**On main (0.1.15 at last read):** core (`ErrorReport`, `Conversion`), ggml (Q8_0, Q4_0), kquants (Q2_K..Q6_K), nf4
(+ double quantization), awq (unpack/pack/repack), lora (merge), CI differential vs llama.cpp (ubuntu+windows,
Q8_0..Q6_K), `docs/PATENT-SWEEP.md` (22 patents read, all clear), CHANGELOG, roadmap current. Branch protection: 5
required checks (gates ubuntu/windows, deny, audit + llama-cpp ubuntu/windows added).

**In flight (rebase onto main on the PM's merge ping, one at a time; the PM allocates versions):**
- #19 `feat/lora-quantized-base` (worktree `ml-convertant-wt-req`): LoRA merge into a quantized base, 0.1.16.
- #20 `feat/mlmf-glue` (worktree `ml-convertant-wt-glue`): `ml-convertant-mlmf`, GGUF in/out, 0.1.17.
Open roadmap rows after those: none. Ideas, not scheduled (new rows are throttled): safetensors glue, NF4/AWQ bases in
the LoRA-into-quantized merge, GPTQ repack, calibration (re-sweep patents first: US12099930B2 H9), more K-quant
formats (IQ*), MLMF-lane asks.

**Facts worth keeping:** Q8_0..Q6_K are byte-identical to ggml's own code compiled from llama.cpp `9d57ce456c`
(harnesses `tools/golden/*_reference.c`, built with -DNDEBUG for the K-quants, MSVC locally, gcc in CI). NF4 encoder
reproduces a real bitsandbytes checkpoint byte for byte. AWQ repack is identical to Fuel's, NOT checked against a real
AWQ checkpoint. LoRA merge is NOT compared with PEFT-merged weights. The glue adds `general.quantization_version` = 2
only when it converted something and the input lacked it.

**Patent sweep:** first record done; re-read triggers are listed in `docs/PATENT-SWEEP.md`. Google Patents throttles
after a few queries (single pages via C:\Projects\tools\patent-claims.py usually answer); USPTO image PDFs via
WebFetch downloadPdf + PyMuPDF work for US documents. Fuel's kernels are exposed to the Nvidia pair
(US12045307B2 claims 1/14/17, US12412082B2), board 307.

**Weekly sweep (2026-10-26):** issues open 0, closed 0, outsiders answered 0; roadmap open 0 after #19/#20 merge.
Next sweep due ~2026-11-02.

**Shell traps:** Bash tool breaks on big quoted heredocs containing apostrophes (write files with the Write tool, run
Python from a file); `gh pr create` can 504: re-check `gh pr list --head` before retrying; worktrees:
`ml-convertant-wt-*` beside the primary checkout, which stays on `main`.
