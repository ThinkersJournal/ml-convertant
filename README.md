# ML-ConvertAnt

Conversions of machine learning model data: quantization and dequantization, packing and
unpacking, merging (such as LoRA), each reporting its measured error.

MLMF reads and writes model files and does not transform the data inside them; this project does.

## Hello: quantize a GGUF

```rust
use ml_convertant::mlmf::{Outcome, Plan, Target, convert_gguf};

let out = convert_gguf(&std::fs::read("in.gguf")?, &Plan::new(Target::Q8_0))?; // or Target::Q4_0
std::fs::write("out.gguf", &out.bytes)?;
for t in &out.tensors {
    if let Outcome::Converted { report, .. } = &t.outcome {
        println!("{}: max error {:.3e}, rmse {:.3e}", t.name, report.max_abs, report.rmse);
    }
}
```

Every tensor of rank two or more with whole 32-element rows is converted; everything else is carried over
unchanged and listed with the reason. The error is measured against the source values, never assumed. A runnable
version: `cargo run --release --example quantize_gguf -- in.gguf out.gguf q4_0`.

| crate | status |
|---|---|
| `ml-convertant-core` | `ErrorReport` and the `Conversion` trait (0.1.1) |
| `ml-convertant-ggml` | Q8_0 and Q4_0, bit-exact vs llama.cpp `9d57ce456c` (0.1.3) |
| `ml-convertant-nf4` | NF4 encode/decode and double quantization, byte-exact vs real bitsandbytes output (0.1.5) |
| `ml-convertant-awq` | AWQ unpack/pack/repack, lossless, identical to Fuel's repack (0.1.6) |
| `ml-convertant-gptq` | GPTQ (AutoGPTQ v1, 4-bit) unpack/pack/dequantize and conversion to the AWQ layout, lossless (unreleased) |
| `ml-convertant-calibrate` | Round-to-nearest and AWQ-style scale-search quantization into the AWQ layout with weight and output error reports (search unreleased) |
| `ml-convertant-kquants` | Q2_K to Q6_K, bit-exact vs ggml (0.1.13) |
| `ml-convertant-mlmf` | GGUF or safetensors in, GGUF out: convert selected tensors with measured error |
| `ml-convertant-lora` | Offline LoRA merge with a proven error bound, into f32, a quantized, NF4 or AWQ base, with adjustable strength |
| `ml-convertant-names` | LoRA adapter tensor names, PEFT <-> GGUF, both directions; unmeasured names refused (unreleased) |

Design: `docs/DESIGN.md`. Licence: MIT OR Apache-2.0.

**Calibrate verification limits:** `ml-convertant-calibrate` is checked against the LoRA crate's AWQ requantizer and hand-worked groups. **Not compared with a real AWQ checkpoint**; the scale search is re-derived from the paper, not compared with AutoAWQ's scales, and output error is measured, not bounded.

**GPTQ verification limits:** `ml-convertant-gptq` is checked against AutoGPTQ's documented packing re-implemented in its tests. **No real GPTQ checkpoint's bytes have been checked**; only 4-bit, and activation-order layers are refused by the AWQ conversion.

**AWQ verification limits:** `ml-convertant-awq` is checked against Fuel's repack output and AutoAWQ's
documented layout on seeded random layers. **No real AWQ checkpoint's bytes have been checked.**
