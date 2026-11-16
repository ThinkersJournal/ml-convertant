# HANDOFF (lane ml-convertant)

State 2026-11-16. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.36 after #38; 0.1.24 published and tagged by the PM):** core, ggml (Q8_0, Q4_0, IQ4_NL), kquants (Q2_K..Q6_K), nf4, awq, gptq (AutoGPTQ v1 4-bit, GPTQ to AWQ
layout), calibrate (RTN and search_awq), lora (merge into f32 / ggml / NF4 / AWQ bases), mlmf glue (GGUF or safetensors in, GGUF out), names; CI differential vs llama.cpp;
`docs/PATENT-SWEEP.md`, `docs/IQ-DESIGN.md`, `docs/IQ-PATENT-SEARCH-2026-11-16.md` (board 307 ruled C: no integer dot kernel here). Rows 1 to 17 done.

**In flight, row 18 (IQ* formats, converters only):** IQ4_XS PR on branch `iq4-xs`, worktree `ml-convertant-wt-iq4xs`, 0.1.37 proposed (PM allocates); byte-identical to ggml `quantize_iq4_xs`,
16 mutations caught. Next: IQ3_S (vendor `iq3s_grid` with ggml's licence text and commit pin in THIRD-PARTY-NOTICES, verify our copy byte for byte against the origin).
Fuel's targets: Iq4Nl/Iq4Xs/Iq3S in GGUF on-disk layout.
**RELEASE GATE: no published version ships IQ* until the CPC/assignee patent database search is done (routed by the PM, not reported finished) or CireSnave decides.**
US12524655B2 (STMicro, lattice VQ) read clear for fixed grids; CONDITION: never build an activation-approximating optimizer with a lattice-distance regularizer.

**Facts worth keeping:** Q8_0..Q6_K, IQ4_NL, IQ4_XS byte-identical to ggml's own code (llama.cpp `9d57ce456c`). NF4 reproduces a real bitsandbytes checkpoint.
AWQ pack order checked against dense Qwen3-0.6B (not AutoAWQ code); GPTQ against one AutoRound-exported checkpoint (act-order unchecked); search_awq not checked against AutoAWQ or real activations. LoRA merge NOT compared
with PEFT-merged weights. IQ* not compared with Fuel's `to_float` or a real IQ file. One version for every crate and path-dep minimum.

**Patent sweep:** calibration re-sweep (#32) and the IQ re-sweep (#37) were thin: web queries, no patent database. Re-read triggers in `docs/PATENT-SWEEP.md`. Google Patents throttles
(use C:\Projects	ools\patent-claims.py). **Weekly sweep:** last 2026-11-12; next due ~2026-11-19.

**Reference harness (WSL):** build `tools/golden/<fmt>_reference.c` with `wsl.exe -d Ubuntu-24.04` gcc against a llama.cpp checkout at the pinned commit (`-O2 -DNDEBUG -ffp-contract=off`); put commands in a script file and run it with `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/...` (inline quoting loses `$vars`; `/tmp` does not persist). MSVC vcvars hangs.

**Shell traps:** big quoted heredocs with apostrophes break Bash (write files with Write, run Python from a file in the scratchpad); worktree files may be CRLF
(normalize in Python); `git add -A` can commit conflict markers (`git grep '<<<<<<<'` before pushing); `gh pr create` can 504 (re-check `gh pr list --head`);
never grep across `C:\Projects`; background commands past their timeout need TaskStop; worktrees `ml-convertant-wt-*` beside the primary, which stays on `main`; repo is ThinkersJournal/ml-convertant.
