# Roadmap

Order follows `docs/DESIGN.md` §4. Status as of 2026-10-25.

| # | Row | Status |
|---|-----|--------|
| 0 | Setup, design page | this PR (#1) |
| 1 | CI on hosted runners (merged #2) | done |
| 2 | Workspace skeleton + `ml-convertant-core` (`ErrorReport`, `Conversion`) | after 1 |
| 3 | Keystone: f32 <-> Q8_0, bound + llama.cpp golden vectors | after 2 |
| 4 | Q4_0 | |
| 5 | NF4 (against the bnb fixture) | |
| 6 | AWQ unpack/repack | |
| 7 | LoRA merge | |
| 8 | mlmf glue (read -> convert -> write) | |

Candidates, unscheduled: K-quants, GPTQ repack, calibration (re-home the code removed from mlmf).
