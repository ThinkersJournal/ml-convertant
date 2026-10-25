# ml-convertant-ggml

ggml block formats, encoded from f32 and decoded back, each with a measured `ErrorReport`.
Currently **Q8_0** (32 elements per block: f16 scale + 32 i8, 34 bytes).

Our own implementation of the published layout, verified byte-for-byte against llama.cpp's
`quantize_row_q8_0_ref` / `dequantize_row_q8_0` at ggml-org/llama.cpp
`9d57ce456c94d241dde672b2db9cf18879766568` (MIT). See `THIRD-PARTY-NOTICES.md` in the repository root.
Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).
