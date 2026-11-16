# HANDOFF (lane ml-convertant)

State 2026-11-16. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.39 after #41; 0.1.24 published and tagged by the PM):** core, ggml (Q8_0, Q4_0, IQ4_NL, IQ4_XS, IQ3_S), kquants (Q2_K..Q6_K), nf4, awq, gptq, calibrate, lora, mlmf glue, names; CI differential vs llama.cpp; patent docs. Rows 1 to 17 done; row 18 encoders done (IQ3_S: ggml bug on exactly-zero sub-blocks kept our correct placement, upstream llama.cpp#32401 open, no reply 2026-11-16).

**Row 18 continued (PM go 2026-11-16: IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S as SEPARATE PRs, DECODE ONLY, no encoders):** IQ3_XXS = branch `iq3-xxs`, worktree `ml-convertant-wt-iq3xxs`, 0.1.40 proposed (tables in `iq_tables.rs` via `tools/golden/gen_iq_tables.py`, generic `tools/golden/iq_dequant_reference.c <format>` and `gen_iq_blocks.py`; extend both per format). IQ2_XXS = branch `iq2-xxs` (worktree `ml-convertant-wt-iq2xxs`, 0.1.41 proposed, stacked on iq3-xxs until #42 merges, PR base then retargets). Real-file IQ3_XXS check (PM yes 2026-11-16): file `smollm2-135m-instruct-iq3_xxs-imat.gguf` from hf nns-hashtags/SmolLM2-135M-Instruct-IQ3_XXS-GGUF, Apache-2.0 (base SmolLM2), 86958272 bytes, sha256 603a9e08e15295c0b1ed1501df1da261fd211b46ac0a50a9cdd029356d1c9794, in C:/Models/gguf-corpus/iq-extra (30 IQ3_XXS + 180 IQ4_NL): extend tests/real_iq_files.rs and gen_real_iq_dequant.py (type 18, 256, 98; local-only, 87MB not in CI) AFTER #42 merges. Next: IQ2_XS (iq2xs_grid u64 x512) (iq2xs_grid u64 x512), IQ2_S (iq2s_grid u64 x1024). Each: tables into gen_iq_tables.py, decode fn, committed random-block vector, CI step, THIRD-PARTY-NOTICES, CHANGELOG, bump. Also a real-file check of each if a small public GGUF ships it (none local: C:/Models has IQ3_XS/IQ4_XS only).
**RELEASE GATE: no published version ships IQ* until the CPC/assignee patent database search AND the family check are done (routed by the PM) or CireSnave decides.** US12524655B2 CONDITION: never build an activation-approximating optimizer with a lattice-distance regularizer.

WAIT: PM merge of the IQ3_XXS PR | check: gh pr list --repo ThinkersJournal/ml-convertant --head iq3-xxs --state merged --json number --jq length | wake: 2026-11-16T22:00:00Z
WAIT: maintainer reply on llama.cpp#32401 | check: gh issue view 32401 --repo ggml-org/llama.cpp --json comments --jq '.comments|length' | wake: 2026-11-18T18:00:00Z
WAIT: weekly patent re-sweep due | check: git log -1 --format=%cs -- docs/PATENT-SWEEP.md | wake: 2026-11-19T12:00:00Z

**Facts worth keeping:** Q8_0..Q6_K, IQ4_NL, IQ4_XS, IQ3_S byte-identical to ggml's own code (llama.cpp `9d57ce456c`). NF4 reproduces a real bitsandbytes checkpoint.
AWQ pack order checked against dense Qwen3-0.6B (not AutoAWQ code); GPTQ against one AutoRound-exported checkpoint (act-order unchecked); search_awq not checked against AutoAWQ or real activations. LoRA merge NOT compared
with PEFT-merged weights. IQ4_NL, IQ4_XS, IQ3_S DECODE checked on 420 real tensors of two SmolLM2-135M IQ GGUFs vs ggml and vs Fuel's `to_float` (fuel-quantized 0.47.32; 0.49.x unpublished, IQ source identical) in `tests/real_iq_files.rs`; ENCODE on real weights unchecked. One version for every crate and path-dep minimum.

**Patent sweep:** calibration re-sweep (#32) and the IQ re-sweep (#37) were thin: web queries, no patent database. Re-read triggers in `docs/PATENT-SWEEP.md`. Google Patents throttles
(use C:\Projects	ools\patent-claims.py). **Weekly sweep:** last 2026-11-12; next due ~2026-11-19.

**Next after the real-file PR merges:** IQ2_* and IQ3_XXS as separate PRs, dequantize/convert only (PM 2026-11-16); gate unchanged. llama.cpp pinned checkout is `~/llama.cpp` in WSL (`/root/llama.cpp`).

**Reference harness (WSL):** build `tools/golden/<fmt>_reference.c` with `wsl.exe -d Ubuntu-24.04` gcc against a llama.cpp checkout at the pinned commit (`-O2 -DNDEBUG -ffp-contract=off`); put commands in a script file and run it with `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/...` (inline quoting loses `$vars`; `/tmp` does not persist). MSVC vcvars hangs.

**Shell traps:** big quoted heredocs with apostrophes break Bash (write files with Write, run Python from a file in the scratchpad); worktree files may be CRLF
(normalize in Python); `git add -A` can commit conflict markers (`git grep '<<<<<<<'` before pushing); `gh pr create` can 504 (re-check `gh pr list --head`);
never grep across `C:\Projects`; background commands past their timeout need TaskStop; worktrees `ml-convertant-wt-*` beside the primary, which stays on `main`; repo is ThinkersJournal/ml-convertant.
