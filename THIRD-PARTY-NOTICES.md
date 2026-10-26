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

Each future entry must give origin (URL + commit), licence, what we copied or derived, and which
files use it. See `docs/DESIGN.md` §5.
