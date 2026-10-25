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
  order, and the double-quantized absmax decode. `ml-convertant-nf4` is our own Rust implementation;
  no bitsandbytes source is copied. The codebook is checked equal to the checkpoint's own `quant_map`.

## Qwen3-0.6B / unsloth bnb fixture (Apache-2.0)

- Origin: `Qwen/Qwen3-0.6B` rev `c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0, Copyright 2024
  Alibaba Cloud) and its bitsandbytes quantization `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` rev
  `e86f5289a2ca40ab7114071959dc2079cf97cb3c` (Apache-2.0).
- Used by: `crates/ml-convertant-nf4/tests/fixtures/bnb_double_quant/` (a 16-row excerpt of one weight
  matrix in quantized and original form, modified as listed in that directory's `PROVENANCE.md`).
  Copied from the Fuel repository's identical fixture (`fuel-tensor/tests/fixtures/bnb_double_quant/`,
  origin/main, 2026-10-25); the SHA-256 digests in `PROVENANCE.md` were re-checked against our copies.
  The licence text is `LICENSE-Qwen3-0.6B-Apache-2.0.txt` beside it.

## half (MIT OR Apache-2.0)

Dependency of `ml-convertant-ggml` (and dev-dependency of the NF4 tests) for f16/bf16 conversion, from crates.io.

## mlmf (MIT OR Apache-2.0)

Dev-dependencies of `ml-convertant-nf4` tests (`mlmf-bnb`, `mlmf-core`, `mlmf-safetensors`, tilde-pinned, crates.io) to read the fixture.

Each future entry must give origin (URL + commit), licence, what we copied or derived, and which
files use it. See `docs/DESIGN.md` §5.
