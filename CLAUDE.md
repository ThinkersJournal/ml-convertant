# ML-ConvertAnt — agent working agreement

Inherits `C:\Projects\CLAUDE.md` (and `AGENT-ONBOARDING.md`); this file adds, never relaxes.

**What this is.** Transforms the DATA inside machine learning models: quantize/dequantize,
pack/unpack/repack, merge (LoRA), each with MEASURED error. Charge, CireSnave 2026-10-25 (decision 275,
first-hand): *"We should have a project for those ML conversions."*

**Boundary with MLMF.** MLMF reads and writes model FILES and only writes data already encoded
(CireSnave 2026-10-24: computing on a model's values is out of MLMF). We do the encoding. File I/O
glue, when added, depends on `mlmf-*` by tilde version from crates.io; the numeric crates take no mlmf
dependency. Deciding what a weight MEANS stays with the consumer (Fuel).

**Rules specific to this repo**
- Every conversion returns its error (`ErrorReport`, `docs/DESIGN.md` §3). A conversion with no
  reported error does not merge.
- Bit-exact against a reference where one exists (llama.cpp, bitsandbytes, AutoAWQ); otherwise a stated
  bound that a test asserts.
- Provenance (`docs/DESIGN.md` §5): no third-party file, table or vector without its origin, licence
  and credit in `THIRD-PARTY-NOTICES.md` in the same PR.
- crates.io versions only; one version number across all crates in the workspace.
- Never Opus; pin `--model sonnet` on every launch and subagent.
- Hosted-runner CI only for now; a test needing a GPU or llama.cpp binary goes on the self-hosted
  runner per `CIRESNAVE-EXPECTATIONS.md` 5.1g/5.1h, never `#[ignore]`.

Roadmap: `ROADMAP.md`. Session state: `HANDOFF.md`.
