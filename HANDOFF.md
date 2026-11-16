# HANDOFF (lane ml-convertant)

State 2026-11-16. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.39 after #41; 0.1.24 published and tagged by the PM):** core, ggml (Q8_0, Q4_0, IQ4_NL, IQ4_XS, IQ3_S), kquants (Q2_K..Q6_K), nf4, awq, gptq, calibrate, lora, mlmf glue, names; CI differential vs llama.cpp; patent docs. Rows 1 to 17 done; row 18 encoders done (IQ3_S: ggml bug on exactly-zero sub-blocks kept our correct placement, upstream llama.cpp#32401 open, no reply 2026-11-16).

**Row 18 continued (PM go 2026-11-16: IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S as SEPARATE PRs, DECODE ONLY, no encoders):** a STACK of PRs, each a branch+worktree `ml-convertant-wt-<x>`: iq3-xxs #42 (0.1.40, in the PM gate) <- iq2-xxs #43 (0.1.41) <- iq2-xs #44 (0.1.42) <- iq2-s (0.1.43, PR to open). As each base merges, retarget/rebase the next onto main (`git rebase --onto origin/main <old base tip>`), then remove worktree+branch per CLAUDE.md s3. Shared machinery: `iq_tables.rs` from `tools/golden/gen_iq_tables.py`, `tools/golden/iq_dequant_reference.c <fmt>`, `gen_iq_blocks.py`, CI step in differential.yml.
**TODO after the stack merges (PM yes 2026-11-16):** real-file decode test for IQ3_XXS: file `smollm2-135m-instruct-iq3_xxs-imat.gguf` from hf nns-hashtags/SmolLM2-135M-Instruct-IQ3_XXS-GGUF, Apache-2.0 (base SmolLM2), 86958272 bytes, sha256 603a9e08e15295c0b1ed1501df1da261fd211b46ac0a50a9cdd029356d1c9794, already in C:/Models/gguf-corpus/iq-extra (30 IQ3_XXS + 180 IQ4_NL): extend `tests/real_iq_files.rs` + `tools/golden/gen_real_iq_dequant.py` (ggml type 18, 256 elems, 98 bytes; Fuel `BlockIq3Xxs`); local-only (87MB, corpus-gated, not fetched in CI). IQ2_XXS/XS/S real files: none found (bartowski has none; HF search empty); ask the PM or skip.
**RELEASE GATE: no published version ships IQ* until the CPC/assignee patent database search AND the family check are done (routed by the PM) or CireSnave decides.** US12524655B2 CONDITION: never build an activation-approximating optimizer with a lattice-distance regularizer.

WAIT: PM merge of the IQ3_XXS PR #42 | check: gh pr list --repo ThinkersJournal/ml-convertant --head iq3-xxs --state merged --json number --jq length | wake: 2026-11-16T22:00:00Z
WAIT: maintainer reply on llama.cpp#32401 | check: gh issue view 32401 --repo ggml-org/llama.cpp --json comments --jq '.comments|length' | wake: 2026-11-18T18:00:00Z
WAIT: weekly check for a small public IQ2_XXS/IQ2_XS/IQ2_S GGUF (imatrix-IQ2 quant of any tiny model suffices; PM 2026-11-16); report in the sweep as 'random-block bit-identity to ggml 9d57ce456c only; no real IQ2_* file tested' until found | check: hf search IQ2_XXS / IQ2_XS / IQ2_S GGUF tiny models (hf_fs ls repos), then extend tests/real_iq_files.rs | wake: 2026-11-19T12:00:00Z
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
