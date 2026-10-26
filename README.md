# ML-ConvertAnt

Conversions of machine learning model data: quantization and dequantization, packing and
unpacking, merging (such as LoRA), each reporting its measured error.

MLMF reads and writes model files and does not transform the data inside them; this project does.

| crate | status |
|---|---|
| `ml-convertant-core` | `ErrorReport` and the `Conversion` trait (0.1.1) |
| `ml-convertant-ggml` | Q8_0 and Q4_0, bit-exact vs llama.cpp `9d57ce456c` (0.1.3) |
| `ml-convertant-nf4` | NF4 encode/decode and double quantization, byte-exact vs real bitsandbytes output (0.1.5) |
| `ml-convertant-awq` | AWQ unpack/pack/repack, lossless, identical to Fuel's repack (0.1.6) |
| `ml-convertant-kquants` | Q2_K to Q6_K, bit-exact vs ggml (0.1.13) |
| `ml-convertant-mlmf` | GGUF in, GGUF out: convert selected tensors with measured error (0.1.17) |
| `ml-convertant-lora` | Offline LoRA merge with a proven error bound (0.1.9) |

Design: `docs/DESIGN.md`. Licence: MIT OR Apache-2.0.

**AWQ verification limits:** `ml-convertant-awq` is checked against Fuel's repack output and AutoAWQ's
documented layout on seeded random layers. **No real AWQ checkpoint's bytes have been checked.**
