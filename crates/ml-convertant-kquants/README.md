# ml-convertant-kquants

ggml K-quants (super-blocks of 256 elements with quantized scales and minimums), encoded from f32 and
decoded back, each with a measured `ErrorReport`. Currently **Q4_K** (144 bytes per super-block).

Our own Rust implementation of the algorithm published in ggml's `ggml-quants.c`
(`quantize_row_q4_K_ref`, `make_qkx2_quants`, `dequantize_row_q4_K`) at ggml-org/llama.cpp
`9d57ce456c94d241dde672b2db9cf18879766568` (MIT), operation for operation in f32 so the bytes are
identical; verified byte-for-byte against that code compiled unmodified. The translation follows the
MIT-licensed source closely and credits it in `THIRD-PARTY-NOTICES.md` in the repository root.
Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).
