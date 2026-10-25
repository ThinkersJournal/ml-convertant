# ml-convertant-nf4

bitsandbytes NF4: f32 to packed 4-bit codes plus per-block absmax, and back, with a measured
`ErrorReport`. Our own implementation of the algorithm bitsandbytes documents (`ops.py` at
bitsandbytes `833649043474794b8fe7a4136e0c40faf077b2e0`, MIT), checked against a real bitsandbytes
checkpoint slice (see `THIRD-PARTY-NOTICES.md` in the repository root).
Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).
