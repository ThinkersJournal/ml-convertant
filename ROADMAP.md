# Roadmap

Order follows `docs/DESIGN.md` §4. Status as of 2026-10-25.

| # | Row | Status |
|---|-----|--------|
| 0 | Setup, design page | this PR (#1) |
| 1 | CI on hosted runners (merged #2) | done |
| 2 | Workspace skeleton + `ml-convertant-core` (`ErrorReport`, `Conversion`) | done |
| 3 | Keystone: f32 <-> Q8_0, bound + llama.cpp golden vectors | done |
| 4 | Q4_0 | done |
| 5 | NF4 encode/decode, byte-exact vs the real bnb fixture | done |
| 5b | NF4 double-quant absmax encode (dynamic map), bit-exact vs the fixture | done |
| 6 | AWQ unpack/pack/repack (lossless, identical to Fuel's repack) | PR open |
| 7 | LoRA merge (f32, bounded; vs f64 reference) | PR open |
| 7b | llama.cpp differential on hosted runners (gcc, pinned commit) | PR open (stacked on 4) |
| 8 | mlmf glue (read -> convert -> write) | |
| 9 | Patent sweep: `docs/PATENT-SWEEP.md` (first record 2026-10-26: 22 read, all clear; re-sweep per its re-read triggers) | done (first pass) |
| 10 | K-quants: Q4_K (bit-exact vs ggml) | PR open |
| 10b | K-quants: Q5_K, Q6_K, Q2_K, Q3_K (one row each) | next |
| 11 | LoRA merge into a quantized base (dequantize, merge, requantize; error of the whole chain vs an f64 reference) | after 7 |

Candidates, unscheduled: K-quants, GPTQ repack, calibration (re-home the code removed from mlmf).
