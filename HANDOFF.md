# HANDOFF (lane ml-convertant)

State 2026-11-15. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.34 at af9c66e; 0.1.24 published and tagged by the PM):** core, ggml (Q8_0, Q4_0), kquants (Q2_K..Q6_K), nf4, awq, gptq (AutoGPTQ v1 4-bit, GPTQ to AWQ
layout), lora (merge into f32 / ggml / NF4 / AWQ bases), mlmf glue (GGUF or safetensors in, GGUF out), names; CI differential vs llama.cpp;
`docs/PATENT-SWEEP.md` (board 307 ruled C: no integer dot kernel here). Rows 1 to 17 done.

**In flight (PM go 2026-11-16, row 18):** IQ* formats. This PR is the thin patent re-sweep (ideas 19 to 21, H23, named-unread) plus `docs/IQ-DESIGN.md`; branch `iq-sweep`,
worktree `ml-convertant-wt-iqsweep`, 0.1.35 (PM allocates). Next: IQ4_NL then IQ4_XS (golden harness against llama.cpp `9d57ce456c`), then IQ3_S (vendor `iq3s_grid` with provenance). Fuel's targets:
Iq4Nl/Iq4Xs/Iq3S in GGUF on-disk layout. Converters only; no kernel (board 307 C). Row 17 done (RTN #33, search #34, real tests #35/#36). Lightbulb and fuel are not on Synapse; the PM relays.

**Facts worth keeping:** Q8_0..Q6_K byte-identical to ggml's own code (llama.cpp `9d57ce456c`). NF4 reproduces a real bitsandbytes checkpoint.
AWQ pack order checked against dense Qwen3-0.6B (not AutoAWQ code); GPTQ against one AutoRound-exported checkpoint (act-order unchecked); search_awq not checked against AutoAWQ or real activations. LoRA merge NOT compared
with PEFT-merged weights. One version for every crate and path-dep minimum.

**Patent sweep:** calibration re-sweep (#32) was thin: two web queries, no patent database. Re-read triggers in `docs/PATENT-SWEEP.md`. Google Patents throttles
(use C:\Projects\tools\patent-claims.py). **Weekly sweep:** last 2026-11-12; next due ~2026-11-19.

**Shell traps:** big quoted heredocs with apostrophes break Bash (write files with Write, run Python from a file in the scratchpad); worktree files may be CRLF
(normalize in Python); `git add -A` can commit conflict markers (`git grep '<<<<<<<'` before pushing); `gh pr create` can 504 (re-check `gh pr list --head`);
never grep across `C:\Projects`; worktrees `ml-convertant-wt-*` beside the primary, which stays on `main`; repo is ThinkersJournal/ml-convertant.
