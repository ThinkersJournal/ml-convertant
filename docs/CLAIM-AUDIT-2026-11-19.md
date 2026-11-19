# Claim-to-command audit, 2026-11-19

Task from pm-kernels (2026-11-19): one command per headline claim in README, ROADMAP and CHANGELOG, VERIFIED or
UNVERIFIED, wrong claims replaced. Ref: branch `iq2-xxs-real` at the commit that carries this file (parent `313a514`), llama.cpp pin `9d57ce456c`.
Full run: `MLCONV_REQUIRE_CORPUS=1 cargo test --workspace --release --locked` on that tree, exit 0, **0 SKIPPED** (so every
corpus-gated test ran), all test binaries ok (including `golden_*`, `real_*`, `peft_merge`, `bnb_fixture`, `fuel_golden`).
Positive control for "the goldens can fail": one byte of `iq4_nl.deq.f32` flipped, `golden_iq4_nl` then FAILED on
`decode_is_bit_identical` (the other four passed), file restored, tree clean.

| Claim (where) | Command | Result |
|---|---|---|
| Hello example converts a GGUF with per-tensor error (README) | `cargo run --release --example quantize_gguf -- SmolLM2-135M-Instruct-f16.gguf out.gguf q4_0` | VERIFIED: prints max error / rmse per tensor, writes 77571136 bytes. Control: an empty input path gives `Os NotFound`; a GGUF v1 file gives `UnsupportedVersion`; a Q8_0 input converts nothing (dense only, as documented) |
| ggml Q8_0, Q4_0, IQ4_NL, IQ4_XS, IQ3_S bit-exact vs llama.cpp `9d57ce456c` (README, CHANGELOG) | `cargo test -p ml-convertant-ggml` (`golden*`) | VERIFIED against committed vectors; regeneration from the pinned reference is CI's job (`differential` jobs), not re-run here. IQ3_S exception (exactly-zero sub-block) is documented and tested |
| IQ4_NL/IQ4_XS/IQ3_S encode byte-identical to ggml on real weights | `cargo test --test real_iq_encode` | VERIFIED (16 rows each of 16 Qwen3-0.6B tensors, no imatrix) |
| IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S decode bit-identical on random blocks | `cargo test -p ml-convertant-ggml --test golden_iq2_s --test golden_iq2_xs --test golden_iq2_xxs --test golden_iq3_xxs` | VERIFIED |
| Real-file decode vs ggml and Fuel (IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS) | `cargo test --test real_iq_files` | VERIFIED (corpus). No real IQ2_XS / IQ2_S file: UNVERIFIED beyond random blocks, stated as such |
| NF4 byte-exact vs real bitsandbytes output | `cargo test -p ml-convertant-nf4 --test bnb_fixture` | VERIFIED (7 tests) |
| AWQ lossless, identical to Fuel's repack | `cargo test -p ml-convertant-awq --test fuel_golden` | VERIFIED |
| AWQ pack order matches a real checkpoint | `cargo test --test real_checkpoints awq_pack_order_against_dense` | VERIFIED (median row corr 0.97 to 0.995; identity order fails per CHANGELOG 0.1.34, not re-mutated here) |
| GPTQ unpack/pack/dequantize, lossless AWQ conversion | `cargo test -p ml-convertant-gptq` and `gptq_*` in `real_checkpoints` | VERIFIED; act-order on a real checkpoint added today, AutoGPTQ's own dequant NOT run |
| K-quants Q2_K..Q6_K bit-exact vs ggml | `cargo test -p ml-convertant-kquants --test golden` | VERIFIED against committed vectors (15 tests) |
| LoRA merge has a proven bound; matches PEFT | `cargo test -p ml-convertant-lora`; `cargo test --test peft_merge` | VERIFIED (26 unit tests; PEFT 0.22.0 on synthetic adapters, layer-0 v_proj first 32 rows, within 2x bound) |
| Names PEFT <-> GGUF | `cargo test -p ml-convertant-names` | VERIFIED (7 unit tests + 2 doc tests) |
| Calibrate RTN / search, error reported not bounded | `cargo test -p ml-convertant-calibrate` | VERIFIED as stated (16 tests); "not compared with AutoAWQ" stands |
| README: gptq and names are "(unreleased)" | download `ml-convertant-gptq-0.1.31.crate` and `-names-0.1.31.crate` from static.crates.io, grep `to_awq_logical`, `peft_to_gguf` | **WRONG, replaced**: both are in 0.1.31 (found); labels now say the version that added them. Control: `search_awq` is NOT in the 0.1.31 calibrate crate, `quantize_iq*` NOT in the 0.1.31 ggml crate, so "search unreleased" and the IQ "unreleased" notes stand |
| README: "No real AWQ checkpoint's bytes have been checked" | `awq_real_checkpoint` passes (196 layers) | **WRONG, replaced** (contradicted 0.1.33/0.1.34 CHANGELOG and the test) |
| README table omitted the IQ3_XXS / IQ2_* decoders | read vs `crates/ml-convertant-ggml/src` | **INCOMPLETE, added** |
| ROADMAP/HANDOFF: "0.1.24 published" | `curl https://crates.io/api/v1/crates/<crate>` for all 11 crates; `git ls-remote --tags origin v0.1.31` | **STALE, replaced**: max_version is 0.1.31 for all 11 crates, tag v0.1.31 on origin |
| Release gate: nothing IQ* ships | 0.1.31 crate has no `quantize_iq*`/`dequantize_iq*` | VERIFIED (nothing later than 0.1.31 is published) |

Not audited: patent-sweep findings (a reading, not a claim about code), CI job behaviour (needs the forge), crates.io
`ml-convertant` facade re-exports.

## Standing UNVERIFIED list (pm-kernels 2026-11-19)

1. Golden vectors regenerated from the pinned llama.cpp `9d57ce456c` (done by CI `differential` jobs and the local WSL harness, not re-run in this audit).
2. CI job behaviour, including `corpus.yml` (the self-hosted runner is not registered, so that job has never run).
3. AutoGPTQ's and AutoAWQ's own output (our checks are against dense weights and documented layouts, not their code).
