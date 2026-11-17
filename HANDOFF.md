# HANDOFF (lane ml-convertant)

State 2026-11-16. FULL SPEED (THROTTLE.md). Box-memory rule: CARGO_BUILD_JOBS=2, one cargo build at a time.

**On main (0.1.44 after #46, 6af06bd; 0.1.24 published and tagged by the PM):** core, ggml (Q8_0, Q4_0, IQ4_NL, IQ4_XS, IQ3_S encode/decode; IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S DECODE only), kquants (Q2_K..Q6_K), nf4, awq, gptq, calibrate, lora, mlmf glue, names; CI differential vs llama.cpp (the llama-cpp contexts now run on every PR: #45 exposed that they never had on the stack); patent docs. Rows 1 to 17 done; row 18 done (IQ3_S: ggml bug on exactly-zero sub-blocks, we keep our correct placement; upstream llama.cpp#32401 open, no reply 2026-11-16). One PR per project per day (CLAUDE.md s9): the PM allocates the version.

**In flight:** #47 (branch `iq3-xxs-real`, worktree `ml-convertant-wt-iq3xxs-real`, 0.1.45 allocated, queued in the PM gate): `.github/workflows/corpus.yml` (runs-on [self-hosted, corpus]; push main + workflow_dispatch(sha) only) and the safetensors pins. #46 (real IQ3_XXS file test) is merged. The corpus job is UNVERIFIED by CI until the PM registers the runner (PM checks restricted account per 5.1h and C:/Models read-only; PM tells me when it is online). Then: dispatch `gh workflow run corpus.yml -f sha=<main sha>`, read the run back, record the result here. Do NOT register a runner myself.

**NO real IQ2_XXS / IQ2_XS / IQ2_S file tested:** "random-block bit-identity to ggml 9d57ce456c only; no real IQ2_* file tested" (say so in every PR and the sweep until a file is found). IQ1_* not scheduled.

**RELEASE GATE: no published version ships IQ* until the CPC/assignee patent database search AND the family check are done (routed by the PM) or CireSnave decides.** US12524655B2 CONDITION: never build an activation-approximating optimizer with a lattice-distance regularizer.

WAIT: PM merge of #47 and corpus runner online | check: gh pr view 47 --json state --jq .state ; then PM message "runner online" | wake: 2026-11-17T18:00:00Z
WAIT: maintainer reply on llama.cpp#32401 | check: gh issue view 32401 --repo ggml-org/llama.cpp --json comments --jq '.comments|length' | wake: 2026-11-18T18:00:00Z
WAIT: weekly check for a small public IQ2_XXS/IQ2_XS/IQ2_S GGUF (imatrix-IQ2 quant of any tiny model suffices; PM 2026-11-16) | check: hf search IQ2_XXS / IQ2_XS / IQ2_S GGUF tiny models (hf_fs ls repos), then extend tests/real_iq_files.rs | wake: 2026-11-19T12:00:00Z
WAIT: weekly patent re-sweep due | check: git log -1 --format=%cs -- docs/PATENT-SWEEP.md | wake: 2026-11-19T12:00:00Z

**Facts worth keeping:** Q8_0..Q6_K, IQ4_NL, IQ4_XS, IQ3_S byte-identical to ggml's own code (llama.cpp `9d57ce456c`); IQ3_XXS/IQ2_* decode bit-identical on random blocks. NF4 reproduces a real bitsandbytes checkpoint.
AWQ pack order checked against dense Qwen3-0.6B (not AutoAWQ code); GPTQ against one AutoRound-exported checkpoint (act-order unchecked); search_awq not checked against AutoAWQ or real activations. LoRA merge NOT compared
with PEFT-merged weights. IQ4_NL, IQ4_XS, IQ3_S DECODE checked on 420 real tensors of two SmolLM2-135M IQ GGUFs vs ggml and vs Fuel's `to_float` (fuel-quantized 0.47.32; 0.49.x unpublished, IQ source identical); ENCODE on real weights unchecked. One version for every crate and path-dep minimum.

**Patent sweep:** calibration re-sweep (#32) and the IQ re-sweep (#37) were thin: web queries, no patent database. Re-read triggers in `docs/PATENT-SWEEP.md`. Google Patents throttles (use C:\Projects\tools\patent-claims.py). **Weekly sweep:** last 2026-11-12; next due ~2026-11-19.

**Reference harness (WSL):** build `tools/golden/<fmt>_reference.c` with `wsl.exe -d Ubuntu-24.04` gcc against a llama.cpp checkout at the pinned commit (`-O2 -DNDEBUG -ffp-contract=off`; checkout is `/root/llama.cpp`); put commands in a script file and run it with `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/...` (inline quoting loses `$vars`; `/tmp` does not persist). Harnesses that write binary files must open with O_BINARY on Windows. Generated Rust (`iq_tables.rs`) carries `#[rustfmt::skip]` so the CI regenerate-and-diff is exact. MSVC vcvars hangs.

**Shell traps:** big quoted heredocs with apostrophes break Bash (write files with Write, run Python from a file in the scratchpad); worktree files may be CRLF
(normalize in Python; a workflow YAML with a raw CR is invalid: validate with PyYAML AND actionlint, diff vs main must show only your lines); `git add -A` can commit conflict markers (`git grep '<<<<<<<'` before pushing); `gh pr create` can 504 (re-check `gh pr list --head`);
never grep across `C:\Projects`; background commands past their timeout need TaskStop; worktrees `ml-convertant-wt-*` beside the primary, which stays on `main`; repo is ThinkersJournal/ml-convertant.
