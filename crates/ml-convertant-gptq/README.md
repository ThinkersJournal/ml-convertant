# ml-convertant-gptq

GPTQ int4 conversions for the AutoGPTQ v1 layout (`qweight [IC/8, OC]` packed along the input
features, `qzeros [G, OC/8]` packed along the output features and stored minus one, `scales [G, OC]`
in f16, optional `g_idx [IC]`): unpack to logical 4-bit codes, pack them back, dequantize, and convert
to the AWQ layout (`ml-convertant-awq`) so a GPTQ checkpoint loads in a W4A16 AWQ path.
Every conversion is lossless and says so in its `ErrorReport` (dequantized weights compared bit for
bit before and after).

Our own implementation of the layout documented by AutoGPTQ (`auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py`,
AutoGPTQ/AutoGPTQ `9f7d37072917`, MIT). See `THIRD-PARTY-NOTICES.md` in the repository root. Part of
[ml-convertant](https://github.com/ThinkersJournal/ml-convertant).

**Scope:** 4-bit only (2, 3 and 8-bit are refused by omission, not handled). Activation-order layers
(`g_idx` not sequential) can be unpacked, packed and dequantized but are **refused** by the AWQ
conversion, which would need the activations permuted too. The Marlin repack lives in baracuda
(`gptq_to_marlin`); this crate does not duplicate it.

**Verification limits:** checked against AutoGPTQ's documented packing, re-implemented independently in the
tests, on seeded random layers and a hand-worked example. No real GPTQ checkpoint's bytes have been checked.
