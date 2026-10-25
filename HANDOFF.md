# HANDOFF (lane ml-convertant)

State 2026-10-25: repo set up from empty; PR "docs: setup and design" open on `docs/setup-and-design`
(worktree `C:\Projects\ml-convertant-wt-design`). Primary checkout stays on `main`.

Next: after the design PR is opened, CI PR (ROADMAP row 1), then stop and report [READY] to the PM.
Needs the PM: merge, version allocation, branch protection once CI is green on main.

PM approved the design (2026-10-25). Versions: #1 = 0.0.1 (bumped), #2 = 0.0.2 (bump Cargo.toml + Cargo.lock
after #1 merges, rebase). Live llama.cpp differential waits until after Q4_0 (self-hosted, board 241).
