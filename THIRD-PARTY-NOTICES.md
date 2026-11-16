# Third-party notices

## ggml / llama.cpp (MIT)

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`.
  Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in `tools/golden/LICENSE-ggml-MIT`.
- What we use: the Q8_0 and Q4_0 block layouts and reference algorithms (`quantize_row_q8_0_ref`,
  `dequantize_row_q8_0`, `quantize_row_q4_0_ref`, `dequantize_row_q4_0` in `ggml/src/ggml-quants.c`). `ml-convertant-ggml` is our own implementation
  of that layout and arithmetic; no ggml source is in the crate.
- Test vectors: `crates/ml-convertant-ggml/tests/golden/q{8,4}_0.{in.f32,out.bin,deq.f32}`. `out.bin` and
  `deq.f32` are the output of ggml's reference functions compiled unmodified from that commit
  (`tools/golden/q{8,4}_0_reference.c` are our harnesses, with three stub symbols for functions it never
  calls); `in.f32` comes from our `tools/golden/gen_q{8,4}_0_inputs.py`. Regenerate with MSVC
  `cl /O2 /fp:precise` plus ggml's `ggml-quants.c` and headers from that commit.

## bitsandbytes (MIT)

- Origin: https://github.com/bitsandbytes-foundation/bitsandbytes, commit
  `833649043474794b8fe7a4136e0c40faf077b2e0`. Licence: MIT (its `LICENSE` at that commit).
- What we use: the NF4 codebook values (`bitsandbytes/functional.py`, `get_4bit_type("nf4")`), and
  the documented algorithm of the reference `quantize_4bit` / `dequantize_4bit` / `dequantize_blockwise`
  in `bitsandbytes/backends/default/ops.py`: block absmax, scaling, nearest-code by midpoints, nibble
  order, and the double-quantized absmax decode; and from `functional.py` the documented
  `create_dynamic_map(signed=True)` construction and `quantize_4bit(compress_statistics=True)` (offset =
  mean absmax, blockwise 8-bit quantization of `absmax - offset`). `ml-convertant-nf4` is our own Rust implementation;
  no bitsandbytes source is copied. The codebook is checked equal to the checkpoint's own `quant_map`.

## Qwen3-0.6B / unsloth bnb fixture (Apache-2.0)

- Origin: `Qwen/Qwen3-0.6B` rev `c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0, Copyright 2024
  Alibaba Cloud) and its bitsandbytes quantization `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` rev
  `e86f5289a2ca40ab7114071959dc2079cf97cb3c` (Apache-2.0).
- Used by: `crates/ml-convertant-nf4/tests/fixtures/bnb_double_quant/` (a 16-row excerpt of one weight
  matrix in quantized and original form, modified as listed in that directory's `PROVENANCE.md`).
  Copied from the Fuel repository's identical fixture (`fuel-tensor/tests/fixtures/bnb_double_quant/`,
  at Fuel origin/main `d02dc578a7a7`, last changed by Fuel commit `54676bbcd4d2`, fetched 2026-10-25;
  Fuel's own origin line for it is its `THIRD-PARTY-NOTICES.md`, which credits the same two upstream repositories
  above under Apache-2.0); the SHA-256 digests in `PROVENANCE.md` were re-checked against our copies.
  The licence text is `LICENSE-Qwen3-0.6B-Apache-2.0.txt` beside it.

## AutoAWQ (MIT)

- Origin: https://github.com/casper-hansen/AutoAWQ, commit `88e4c76b20755db275574e6a03c83c84ba3bece5`.
  Licence: MIT.
- What we use: the documented packing layout and order (`awq/modules/linear/gemm.py`
  `WQLinear_GEMM.from_linear`, `order_map = [0, 2, 4, 6, 1, 3, 5, 7]`) and the unpack/dequantize
  formulation in `awq/utils/packing_utils.py` (`AWQ_REVERSE_ORDER`, `dequantize_gemm`).
  `ml-convertant-awq` is our own implementation; no AutoAWQ source is copied. Its tests re-derive
  the unpacking with AutoAWQ's reverse-order formulation as an independent check.

## AutoGPTQ (MIT)

- Origin: https://github.com/AutoGPTQ/AutoGPTQ, commit `9f7d37072917` (main as read 2026-11-15; the project is
  unmaintained). Licence: MIT, Copyright (c) 2023 William.
- What we use: the documented 4-bit layout in `auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py` (buffer shapes in
  `__init__`, the `pack` method's packing order, `zeros -= 1` on pack and `zeros + 1` masked on read).
  `ml-convertant-gptq` is our own implementation; no AutoGPTQ source is copied. Its tests re-implement `pack`
  independently as the check.
- Test data, not committed: `tests/real_checkpoints.rs` reads a real GPTQ checkpoint from the local corpus
  (`C:/Models/SmolLM2-360M-Instruct-GPTQ-gs64-asym`), which is Intel AutoRound's export in AutoGPTQ v1 format
  (`fbaldassarri/HuggingFaceTB_SmolLM2-360M-Instruct-auto_gptq-int4-gs64-asym` at revision
  `f3e7f293740f7b8c7b88858d4a879a90acf47b67`), against the dense `HuggingFaceTB/SmolLM2-360M-Instruct`. No weight bytes are copied into this repository.

## Fuel awq_repack output (own code, MIT OR Apache-2.0)

- `crates/ml-convertant-awq/tests/golden/fuel_repack.bin` records the output of Fuel's
  `fuel-quantized/src/awq_repack.rs` (Fuel origin/main `d02dc578a7a7`, file last changed by Fuel commit
  `35d8d966f7f1`) on seeded random input; generator and format in `tools/golden/fuel_awq/`. Output data
  only; none of Fuel's code is copied here. Fuel's own origin note for that file: written against
  AutoAWQ's GEMM layout and baracuda's `Int4AwqGemmArgs` documentation.

## PEFT (Apache-2.0)

- Origin: https://github.com/huggingface/peft, commit `7d6315ae910bf62b6044a7ca917bccf5881f5e43`.
  Licence: Apache-2.0.
- What we use: the documented LoRA merge formula and tensor layouts
  (`src/peft/tuners/lora/layer.py`: `get_delta_weight` = `(lora_B.weight @ lora_A.weight) * scaling`,
  `merge` adds it to the base weight; scaling `lora_alpha / r`, or `lora_alpha / sqrt(r)` with
  `use_rslora`). `ml-convertant-lora` is our own implementation; no PEFT source is copied.

## ggml IQ4_NL (MIT), `ml-convertant-ggml`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`: `quantize_iq4_nl`, `quantize_row_iq4_nl_impl`, `best_index_int8`, `dequantize_row_iq4_nl`;
  `ggml/src/ggml-common.h`: `block_iq4_nl`, `kvalues_iq4nl`). Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in
  `tools/golden/LICENSE-ggml-MIT`.
- What we use: the block layout, the sixteen-entry table `kvalues_iq4nl` (`-127 ... 113`, copied as data) and the encoder's algorithm.
  `ml-convertant-ggml`'s `iq4_nl.rs` is our own implementation of that arithmetic; no ggml source file is in the crate.
- Verification of our copy against the origin: the table and the encoder are checked byte for byte by compiling ggml's own functions from that
  commit (`tools/golden/iq4_nl_reference.c`) and comparing: committed vectors `crates/ml-convertant-ggml/tests/golden/iq4_nl*`
  (every one of the sixteen nibble values occurs in them, so every table entry is exercised), and the CI differential on random data.
  The vectors were produced with gcc 13.3 (`-O2 -DNDEBUG -ffp-contract=off`) on Ubuntu 24.04.

## ggml IQ4_XS (MIT), `ml-convertant-ggml`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`: `quantize_iq4_xs`, `quantize_row_iq4_nl_impl` with `QK_K` blocks, `dequantize_row_iq4_xs`;
  `ggml/src/ggml-common.h`: `block_iq4_xs`). Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in
  `tools/golden/LICENSE-ggml-MIT`. The sixteen-entry table is the one of the IQ4_NL section above.
- What we use: the block layout and the encoder's algorithm. `ml-convertant-ggml`'s `iq4_xs.rs` is our own implementation; no ggml source file is in the crate.
- Verification of our copy against the origin: the encoder and decoder are checked byte for byte by compiling ggml's own functions from that
  commit (`tools/golden/iq4_xs_reference.c`) and comparing: committed vectors `crates/ml-convertant-ggml/tests/golden/iq4_xs*`
  (all sixteen nibble values and both ends of the 6-bit scale range occur in them), and the CI differential on random data.
  The vectors were produced with gcc 13.3 (`-O2 -DNDEBUG -ffp-contract=off`) on Ubuntu 24.04.

## ggml IQ3_S (MIT), `ml-convertant-ggml`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`: `quantize_iq3_s`, `quantize_row_iq3_s_impl`, `iq3xs_init_impl`, `iq3_find_best_neighbour`, `dequantize_row_iq3_s`;
  `ggml/src/ggml-common.h`: `block_iq3_s`, the 512-entry table `iq3s_grid`, `kmask_iq2xs`). Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in
  `tools/golden/LICENSE-ggml-MIT`.
- What we use: the block layout, the table `iq3s_grid` (512 `u32`, copied as data into `crates/ml-convertant-ggml/src/iq3s_grid.rs`, a generated file that
  carries this origin in its header) and the encoder's algorithm. `iq3_s.rs` is our own implementation; no ggml source file is in the crate.
- Verification of our copy against the origin: `tools/golden/gen_iq3s_grid.py` extracts `iq3s_grid` from the pinned `ggml-common.h`, checks every entry against
  ggml's own `kgrid_512` (the table `iq3xs_init_impl` builds its encoder grid from) and writes the Rust file; CI re-runs it on the pinned checkout and diffs the result
  against the committed file. The encoder and decoder are checked byte for byte by compiling ggml's own functions from that commit (`tools/golden/iq3_s_reference.c`):
  committed vectors `crates/ml-convertant-ggml/tests/golden/iq3_s*` and the CI differential on random data. The vectors were produced with gcc 13.3
  (`-O2 -DNDEBUG -ffp-contract=off`) on Ubuntu 24.04. The one block shape where we differ from ggml on purpose is recorded in `CHANGELOG.md` and `iq3_s.rs`.

## ggml IQ3_XXS decode (MIT), `ml-convertant-ggml`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`: `dequantize_row_iq3_xxs`; `ggml/src/ggml-common.h`: `block_iq3_xxs`, the tables `iq3xxs_grid` (256 `u32`), `ksigns_iq2xs` (128 `u8`), `kmask_iq2xs` (8 `u8`)). Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in
  `tools/golden/LICENSE-ggml-MIT`.
- What we use: the block layout, the three tables (copied as data into `crates/ml-convertant-ggml/src/iq_tables.rs`, a generated file that carries this origin in its header) and the decode arithmetic. `iq3_xxs.rs` is our own implementation; no ggml source file is in the crate. Decode only: no encoder.
- Verification of our copy against the origin: `tools/golden/gen_iq_tables.py` extracts the tables from the pinned `ggml-common.h` (length checked against the declaration) and writes the Rust file; CI re-runs it on the pinned checkout and diffs the result against the committed file. The decoder is checked bit for bit by compiling ggml's own `dequantize_row_iq3_xxs` (`tools/golden/iq_dequant_reference.c`) on 64 random blocks that use every grid point and every sign pattern: `crates/ml-convertant-ggml/tests/golden/iq3_xxs*`, reproduced by CI. Produced with gcc 13.3 (`-O2 -DNDEBUG -ffp-contract=off`) on Ubuntu 24.04.

## ggml IQ2_XXS decode (MIT), `ml-convertant-ggml`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`: `dequantize_row_iq2_xxs`; `ggml/src/ggml-common.h`: `block_iq2_xxs`, the table `iq2xxs_grid` (256 `u64`), plus `ksigns_iq2xs` and `kmask_iq2xs` as for IQ3_XXS). Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in
  `tools/golden/LICENSE-ggml-MIT`.
- What we use: the block layout, the table (copied as data into `crates/ml-convertant-ggml/src/iq_tables.rs`, generated, carrying this origin in its header) and the decode arithmetic. `iq2_xxs.rs` is our own implementation; no ggml source file is in the crate. Decode only.
- Verification: as for IQ3_XXS above: `tools/golden/gen_iq_tables.py` is re-run by CI on the pinned checkout and diffed against the committed file; ggml's own `dequantize_row_iq2_xxs` (`tools/golden/iq_dequant_reference.c`) is compared bit for bit on 64 random blocks covering every grid point and sign pattern (`crates/ml-convertant-ggml/tests/golden/iq2_xxs*`), reproduced by CI. gcc 13.3 (`-O2 -DNDEBUG -ffp-contract=off`), Ubuntu 24.04.

## ggml IQ2_XS decode (MIT), `ml-convertant-ggml`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`: `dequantize_row_iq2_xs`; `ggml/src/ggml-common.h`: `block_iq2_xs`, the table `iq2xs_grid` (512 `u64`), plus `ksigns_iq2xs` and `kmask_iq2xs` as for IQ3_XXS). Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in
  `tools/golden/LICENSE-ggml-MIT`.
- What we use: the block layout, the table (copied as data into `crates/ml-convertant-ggml/src/iq_tables.rs`, generated, carrying this origin in its header) and the decode arithmetic. `iq2_xs.rs` is our own implementation; no ggml source file is in the crate. Decode only.
- Verification: as for IQ3_XXS above: `tools/golden/gen_iq_tables.py` is re-run by CI on the pinned checkout and diffed against the committed file; ggml's own `dequantize_row_iq2_xs` (`tools/golden/iq_dequant_reference.c`) is compared bit for bit on 256 random blocks covering every grid point and sign pattern (`crates/ml-convertant-ggml/tests/golden/iq2_xs*`), reproduced by CI. gcc 13.3 (`-O2 -DNDEBUG -ffp-contract=off`), Ubuntu 24.04.

## ggml K-quants (MIT), `ml-convertant-kquants`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`ggml/src/ggml-quants.c`, `ggml/src/ggml-common.h`). Licence: MIT, Copyright (c) 2023-2026 The ggml
  authors; text in `tools/golden/LICENSE-ggml-MIT`.
- What we derive: the Q2_K, Q3_K, Q4_K, Q5_K and Q6_K algorithms (`quantize_row_q2_K_ref`, `quantize_row_q3_K_ref`, `dequantize_row_q2_K`, `dequantize_row_q3_K`, `make_q3_quants`, `quantize_row_q4_K_ref`, `quantize_row_q5_K_ref`, `quantize_row_q6_K_ref`, `dequantize_row_q5_K`, `dequantize_row_q6_K`, `make_qx_quants`, `make_qkx2_quants`, `nearest_int`,
  `get_scale_min_k4`, `dequantize_row_q4_K`). `crates/ml-convertant-kquants/src/lib.rs` is a Rust
  translation of that published algorithm that follows the source operation for operation (the bytes must
  be identical), so it is a derivative work of the MIT-licensed source and carries that credit and licence.
- Test vectors: `crates/ml-convertant-kquants/tests/golden/q{2,3,4,5,6}_k.{in.f32,out.bin,deq.f32}`: `out.bin` and
  `deq.f32` come from ggml's functions compiled unmodified from that commit with `tools/golden/q{2,3,4,5,6}_k_reference.c`
  (our harness; built with `-DNDEBUG` as llama.cpp does); `in.f32` from `tools/golden/gen_q{2,3,4,5,6}_k_inputs.py`.
- Not copied: Fuel's `k_quants.rs` (candle-derived, see `docs/DESIGN.md` §5); we translated from ggml directly.

## half (MIT OR Apache-2.0)

Dependency of `ml-convertant-ggml` and `ml-convertant-awq` (and dev-dependency of the NF4 tests) for f16/bf16 conversion, from crates.io.

## mlmf (MIT OR Apache-2.0)

`ml-convertant-mlmf` depends on `mlmf-core`, `mlmf-ggml` and `mlmf-gguf` (tilde-pinned, crates.io) to read and write GGUF.
Dev-dependencies of `ml-convertant-nf4` tests (`mlmf-bnb`, `mlmf-core`, `mlmf-safetensors`, tilde-pinned, crates.io) to read the fixture.

## llama.cpp adapter tensor naming (MIT), `ml-convertant-names`

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`
  (`gguf-py/gguf/tensor_mapping.py`, `convert_lora_to_gguf.py`). Licence: MIT, Copyright (c) 2023-2026 The ggml
  authors; text in `tools/golden/LICENSE-ggml-MIT`.
- What we use: the facts that a Llama-style Hugging Face module maps to a GGUF stem (`self_attn.q_proj` is `attn_q`, ... `lm_head` is
  `output`, `model.embed_tokens` is `token_embd`), that an adapter factor is named `<base>.weight.lora_a` / `.lora_b`, that
  `.base_layer.weight` is skipped, and that `token_embd`'s `lora_a` is transposed. `crates/ml-convertant-names/src/lib.rs` is our own
  table of seven module names and two globals; no llama.cpp source is copied.
- Checked against: real adapter tensor names recorded in MLMF's `lora-measurements.tsv` (a llama GGUF adapter; PEFT Llama, Gemma 2,
  Qwen-style adapters); see the crate docs for what is refused.

Each future entry must give origin (URL + commit), licence, what we copied or derived, and which
files use it. See `docs/DESIGN.md` §5.
