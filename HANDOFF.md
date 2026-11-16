# HANDOFF (lane ml-convertant)

State 2026-11-15. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.31 at ce5843f; 0.1.24 published and tagged by the PM):** core, ggml (Q8_0, Q4_0), kquants (Q2_K..Q6_K), nf4, awq, gptq (AutoGPTQ v1 4-bit, GPTQ to AWQ
layout), lora (merge into f32 / ggml / NF4 / AWQ bases), mlmf glue (GGUF or safetensors in, GGUF out), names; CI differential vs llama.cpp;
`docs/PATENT-SWEEP.md` (board 307 ruled C: no integer dot kernel here). Rows 1 to 16 done; 17 half (RTN).

**In flight:** row 17 calibration, crate `ml-convertant-calibrate` (0.1.31 published, RTN). This PR: `search_awq`, branch `calibrate-search`, worktree
`ml-convertant-wt-search`, 0.1.32 proposed (PM allocates). After it merges: IQ* formats (patent re-sweep first; ask the PM if it finds a row). Lightbulb and fuel are not on Synapse; the PM relays. Converters only; no kernel (board 307 C). Do not combine H9's tail-discard,
moving-average maximum and power-of-two clipping.

**Facts worth keeping:** Q8_0..Q6_K byte-identical to ggml's own code (llama.cpp `9d57ce456c`). NF4 reproduces a real bitsandbytes checkpoint.
AWQ repack identical to Fuel's, NOT checked against a real AWQ checkpoint; GPTQ and calibrate likewise unchecked on real checkpoints. LoRA merge NOT compared
with PEFT-merged weights. One version for every crate and path-dep minimum.

**Patent sweep:** calibration re-sweep (#32) was thin: two web queries, no patent database. Re-read triggers in `docs/PATENT-SWEEP.md`. Google Patents throttles
(use C:\Projects\tools\patent-claims.py). **Weekly sweep:** last 2026-11-12; next due ~2026-11-19.

**Shell traps:** big quoted heredocs with apostrophes break Bash (write files with Write, run Python from a file in the scratchpad); worktree files may be CRLF
(normalize in Python); `git add -A` can commit conflict markers (`git grep '<<<<<<<'` before pushing); `gh pr create` can 504 (re-check `gh pr list --head`);
never grep across `C:\Projects`; worktrees `ml-convertant-wt-*` beside the primary, which stays on `main`; repo is ThinkersJournal/ml-convertant.
