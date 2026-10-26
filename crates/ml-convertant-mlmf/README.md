# ml-convertant-mlmf

Read a GGUF with [MLMF](https://crates.io/crates/mlmf-gguf), convert chosen dense tensors (F32, F16, BF16) to a
ggml quantized type (Q8_0, Q4_0, Q2_K to Q6_K) with `ml-convertant`, and write a new GGUF with MLMF's writer.
MLMF reads and writes the file and never interprets it; the conversion and its measured error are this crate's.

Everything else is carried over untouched: all metadata, every unselected tensor's bytes and declared type.
A file this build cannot fully carry (unrecognized metadata or tensor types, an incomplete key index) is refused
rather than written with things missing. Which tensors to quantize is the caller's policy (`Plan::with_select`);
the default selects every tensor of rank two or more. Tensors whose first dimension is not a whole number of
blocks, or that are not dense F32/F16/BF16, are copied and listed as skipped with the reason.

Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).
