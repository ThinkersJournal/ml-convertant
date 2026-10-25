Generator for `crates/ml-convertant-awq/tests/golden/fuel_repack.bin`.

It runs Fuel's `repack_hf_awq_for_kernel` (fuel-quantized/src/awq_repack.rs, Fuel origin/main
`d02dc578a7a74f823be73ab4e2ffd7db32700cad`, file last changed by Fuel commit
`35d8d966f7f1c5c9de526a8ea7156e7230af7af7`; Fuel's own code, MIT OR Apache-2.0) on seeded random
HuggingFace-layout words and records its output. Only that output is committed; none of Fuel's code
is copied into this repository. To regenerate: make a crate from `Cargo.toml.txt` and `main.rs`, put
that Fuel file at `src/awq_repack.rs`, and run it with the output path as the argument.

File format (little-endian): `u32` case count; per case `u32` ic, oc, group_size, zeros_w, then `u32`
lengths of qweight, qzeros, scales, weight, zeros, scales_out; then the `u32` words of qweight,
qzeros, weight, zeros; then the `u16` f16 bits of scales and scales_out.
