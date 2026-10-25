# Third-party notices

## ggml / llama.cpp (MIT)

- Origin: https://github.com/ggml-org/llama.cpp, commit `9d57ce456c94d241dde672b2db9cf18879766568`.
  Licence: MIT, Copyright (c) 2023-2026 The ggml authors; text in `tools/golden/LICENSE-ggml-MIT`.
- What we use: the Q8_0 block layout and the reference algorithm (`quantize_row_q8_0_ref`,
  `dequantize_row_q8_0` in `ggml/src/ggml-quants.c`). `ml-convertant-ggml` is our own implementation
  of that layout and arithmetic; no ggml source is in the crate.
- Test vectors: `crates/ml-convertant-ggml/tests/golden/q8_0.{in.f32,out.bin,deq.f32}`. `out.bin` and
  `deq.f32` are the output of ggml's reference functions compiled unmodified from that commit
  (`tools/golden/q8_0_reference.c` is our harness, with three stub symbols for functions it never
  calls); `in.f32` comes from our `tools/golden/gen_q8_0_inputs.py`. Regenerate with MSVC
  `cl /O2 /fp:precise` plus ggml's `ggml-quants.c` and headers from that commit.

## half (MIT OR Apache-2.0)

Dependency of `ml-convertant-ggml` for f16 conversion, from crates.io.

Each future entry must give origin (URL + commit), licence, what we copied or derived, and which
files use it. See `docs/DESIGN.md` §5.
