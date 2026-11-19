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
| `ml-convertant-ggml` | Q8_0, Q4_0, IQ4_NL, IQ4_XS and IQ3_S (importance vector optional), bit-exact vs llama.cpp `9d57ce456c` (IQ3_S except blocks with an exactly-zero sub-block, see CHANGELOG; IQ4_NL, IQ4_XS and IQ3_S unreleased: patent gate, see `docs/PATENT-SWEEP.md`) |
| `ml-convertant-nf4` | NF4 encode/decode and double quantization, byte-exact vs real bitsandbytes output (0.1.5) |
| `ml-convertant-awq` | AWQ unpack/pack/repack, lossless, identical to Fuel's repack (0.1.6) |
| `ml-convertant-gptq` | GPTQ (AutoGPTQ v1, 4-bit) unpack/pack/dequantize and conversion to the AWQ layout, lossless (unreleased) |
| `ml-convertant-calibrate` | Round-to-nearest and AWQ-style scale-search quantization into the AWQ layout with weight and output error reports (search unreleased) |
| `ml-convertant-kquants` | Q2_K to Q6_K, bit-exact vs ggml (0.1.13) |
| `ml-convertant-mlmf` | GGUF or safetensors in, GGUF out: convert selected tensors with measured error |
| `ml-convertant-lora` | Offline LoRA merge with a proven error bound, into f32, a quantized, NF4 or AWQ base, with adjustable strength |
| `ml-convertant-names` | LoRA adapter tensor names, PEFT <-> GGUF, both directions; unmeasured names refused (unreleased) |

Design: `docs/DESIGN.md`. Licence: MIT OR Apache-2.0.

**Real-checkpoint test (corpus-gated):** `tests/real_checkpoints.rs` runs real AWQ layers and real dense weights when `C:/Models` exists. Round trips and ranges cannot see a column permutation, so `awq_pack_order_against_dense` correlates dequantized rows with dense Qwen3-0.6B (median 0.97 to 0.995; identity order gives about 0.24 and fails). That verifies the pack order against a real checkpoint, **not against AutoAWQ's code**. Calibrate ran on synthetic activations there.

**Calibrate verification limits:** `ml-convertant-calibrate` is checked against the LoRA crate's AWQ requantizer and hand-worked groups. **Not compared with a real AWQ checkpoint**; the scale search is re-derived from the paper, not compared with AutoAWQ's scales, and output error is measured, not bounded.

**GPTQ verification limits:** `ml-convertant-gptq` is checked against AutoGPTQ's documented packing re-implemented in its tests. One real GPTQ checkpoint (Intel AutoRound's AutoGPTQ-v1 export of SmolLM2-360M, gs64 asymmetric) is checked against dense weights by a corpus-gated test (correlation 0.986 to 0.996; zero offset and nibble order mutations fail); a second, act-order one (TheBloke's AutoGPTQ-made TinyLlama-1.1B-Chat-v1.0, group 128, symmetric, `desc_act=true`) is checked the same way on 3 layers: `g_idx` is non-sequential, the AWQ conversion refuses it, pack(unpack) round-trips, and dequantizing with `g_idx` agrees with dense (median row correlation 0.93 to 0.99; relative RMSE below 0.16 except layer-0 q_proj 0.32 and k_proj 0.45) while ignoring `g_idx` is at least 1.5 times worse; AutoGPTQ's own dequantization was not run; only 4-bit, and activation-order layers are refused by the AWQ conversion.

**AWQ verification limits:** `ml-convertant-awq` is checked against Fuel's repack output and AutoAWQ's
documented layout on seeded random layers. **No real AWQ checkpoint's bytes have been checked.**
