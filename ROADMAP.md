# Roadmap

Order follows `docs/DESIGN.md` §4. Status as of 2026-11-14.

| # | Row | Status |
|---|-----|--------|
| 0 | Setup, design page | done |
| 1 | CI on hosted runners (merged #2) | done |
| 2 | Workspace skeleton + `ml-convertant-core` (`ErrorReport`, `Conversion`) | done |
| 3 | Keystone: f32 <-> Q8_0, bound + llama.cpp golden vectors | done |
| 4 | Q4_0 | done |
| 5 | NF4 encode/decode, byte-exact vs the real bnb fixture | done |
| 5b | NF4 double-quant absmax encode (dynamic map), bit-exact vs the fixture | done |
| 6 | AWQ unpack/pack/repack (lossless, identical to Fuel's repack) | done |
| 7 | LoRA merge (f32, bounded; vs f64 reference) | done |
| 7b | llama.cpp differential on hosted runners (gcc, pinned commit) | done |
| 8 | mlmf glue (read -> convert -> write) | done (GGUF; safetensors later) |
| 9 | Patent sweep: `docs/PATENT-SWEEP.md` (first record 2026-10-26: 22 read, all clear; re-sweep per its re-read triggers) | done (first pass) |
| 10 | K-quants: Q4_K (bit-exact vs ggml) | done |
| 10b | K-quants: Q5_K (bit-exact vs ggml) | done |
| 10c | K-quants: Q6_K (bit-exact vs ggml) | done |
| 10d | K-quants: Q2_K and Q3_K (bit-exact vs ggml) | done |
| 11 | LoRA merge into a quantized base (dequantize, merge, requantize; error of the whole chain vs an f64 reference) | done |
| 12 | `ml-convertant-names`: PEFT <-> GGUF adapter tensor names (ruling 421 Q1, merged #26) | done |
| 13 | Adapter strength: `LoraAdapter::with_strength` (ruling 421 Q2, merged #27) | done |
| 14 | safetensors glue: `convert_safetensors` (safetensors in, GGUF out; PM [TASK] 2026-11-15, merged #29) | done |
| 15 | NF4/AWQ bases for merge-into-quantized (`merge_lora_into_nf4`, `merge_lora_into_awq`) | done |
| 16 | GPTQ: `ml-convertant-gptq` (unpack/pack/dequantize, GPTQ to AWQ layout; Marlin stays in baracuda; merged #31) | done |
| 17 | Calibration: `ml-convertant-calibrate`, RTN baseline then AWQ-style scale search (`docs/CALIBRATION-DESIGN.md`; PM [TASK] 2026-11-15) | done (RTN #33, search #34; real-checkpoint tests #35, #36) |
| 18 | IQ* formats: `IQ4_NL`, `IQ4_XS`, `IQ3_S` encoders and `IQ3_XXS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S` decoders, bit-exact vs ggml (`docs/IQ-DESIGN.md`; PM go 2026-11-16, converters only) | sweep and design merged; IQ4_NL merged; IQ4_XS merged; IQ3_S merged (#40); real-file decode check vs ggml and Fuel merged (#41); IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S decode merged (#45, one PR; random-block bit-identity to ggml); real IQ3_XXS file checked vs ggml and Fuel (0.1.44); NO real IQ2_* file tested (rechecked weekly); IQ1_S and IQ1_M decode (random-block bit-identity to ggml; real IQ1_S TinyLlama file checked vs ggml, 108 tensors; no real IQ1_M file tested; PM [TASK] 2026-11-19) |

**Releases:** 0.1.24 (first; board 307 ruled C, 2026-11-01) and 0.1.31 published and tagged (crates.io latest of every crate, read 2026-11-19; tags v0.1.24, v0.1.31). Nothing later is published. Both 421 rulings are implemented.

Candidates, unscheduled (no row until the PM or CireSnave says go): safetensors glue, NF4/AWQ bases for merge-into-quantized, GPTQ repack, calibration (re-sweep patents first, H9), IQ* K-quant formats, differential for other formats.
