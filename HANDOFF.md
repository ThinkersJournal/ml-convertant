# HANDOFF (lane ml-convertant)

State 2026-10-25: main = 0.0.1 with CI and branch protection. Row 2 (core crate, 0.1.0) in PR from
`feat/core-crate` (worktree `C:\Projects\ml-convertant-wt-core`). Primary checkout stays on `main`.

Next after merge: row 3 keystone, f32 <-> Q8_0 (`ml-convertant-ggml`): bound + llama.cpp golden vectors
at `9d57ce456c`, written fresh (do not copy Fuel's k_quants: candle-derived, see DESIGN §5). One PR per row.
The PM allocates versions at gate time. Live llama.cpp differential waits until after Q4_0.
