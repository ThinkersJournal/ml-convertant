# ml-convertant-lora

Offline LoRA merge: `W' = W + scale * (B @ A)` for a `[out, in]` weight, `A` of shape `[rank, in]` and
`B` of shape `[out, rank]` (the layout of PEFT's `lora_A.weight` / `lora_B.weight`), in f32 with a
fixed accumulation order. The `ErrorReport` compares the result with the same merge evaluated in f64
and rounded to f32, and carries a proven per-tensor bound.

The formula is the one PEFT documents (`peft/tuners/lora/layer.py`, huggingface/peft `7d6315ae9`,
Apache-2.0); no PEFT source is copied. **Verified** against hand-computed cases and the f64 reference;
**not** compared against merged weights produced by PEFT itself.
Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).
