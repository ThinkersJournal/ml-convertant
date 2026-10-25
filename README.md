# ML-ConvertAnt

Conversions of machine learning model data: quantization and dequantization, packing and
unpacking, merging (such as LoRA), each reporting its measured error.

MLMF reads and writes model files and does not transform the data inside them; this project does.

| crate | status |
|---|---|
| `ml-convertant-core` | `ErrorReport` and the `Conversion` trait (0.1.1) |
| `ml-convertant-ggml` | Q8_0 and Q4_0, bit-exact vs llama.cpp `9d57ce456c` (0.1.3) |
| Q4_0, NF4, AWQ repack, LoRA merge | planned, see `ROADMAP.md` |

Design: `docs/DESIGN.md`. Licence: MIT OR Apache-2.0.
