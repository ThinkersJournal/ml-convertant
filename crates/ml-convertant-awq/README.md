# ml-convertant-awq

AWQ int4 (GEMM version) conversions: unpack a HuggingFace `*-AWQ` layer to logical 4-bit codes, pack
codes back the way AutoAWQ does, and repack to the OC-major kernel layout used by W4A16 GEMM kernels.
Every conversion is lossless and says so in its `ErrorReport` (dequantized weights compared
bit for bit before and after).

Our own implementation of the layouts documented by AutoAWQ (`awq/modules/linear/gemm.py`,
`awq/utils/packing_utils.py`, casper-hansen/AutoAWQ `88e4c76b2`, MIT). See `THIRD-PARTY-NOTICES.md` in
the repository root. Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).

**Verification limits:** checked against Fuel's repack output and AutoAWQ's documented layout on seeded
random layers. No real AWQ checkpoint's bytes have been checked.
