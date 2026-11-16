# IQ* formats design note (ROADMAP row 18, proposed)

Status: design only, no code. Lane ml-convertant, 2026-11-16. Patent reading: thin (`PATENT-SWEEP.md` ideas 19 to 21, queries W8 to W10:
no patent database searched, H23 the one patent found). Not legal advice.

## Target set (what Fuel loads)

Fuel 0.49.0 reads `IQ4_NL`, `IQ4_XS` and `IQ3_S` in GGUF's on-disk layout (`fuel-quantized` `iq4.rs`, `iq_grid_types.rs`; the 27B Qwen3.5 file
has IQ4_XS x142, IQ4_NL x9, IQ3_S x2). We write exactly those bytes, so the output of `convert` loads in Fuel and llama.cpp unchanged.
Converters only (board 307 C): encode and decode with measured error; no kernel.

| Type | Block | Bytes | Needs |
| --- | --- | --- | --- |
| IQ4_NL | 32 elements: f16 `d`, 16 bytes of nibbles into `kvalues_iq4nl` | 18 | 16-entry i8 table (ggml-common.h, MIT) |
| IQ4_XS | 256: f16 `d`, u16 `scales_h`, 4 bytes `scales_l`, 128 bytes nibbles | 136 | same table, 6-bit sub-block scales |
| IQ3_S | 256: f16 `d`, 64 `qs`, 8 `qh`, 32 `signs`, 4 `scales` | 110 | the 512-entry `iq3s_grid` (u32 each) and its neighbour map |

## Order (one PR each, bit-exact against llama.cpp `9d57ce456c` compiled in the existing golden harness, as the K-quants are)

1. **IQ4_NL** then **IQ4_XS**: table-based, the encoder is ggml's `quantize_row_iq4_nl_impl` (scale search over the table, optional importance weights). Small.
2. **IQ3_S**: the grid table must be vendored with its origin, licence and credit in `THIRD-PARTY-NOTICES.md` (ggml-common.h, MIT; we do not copy encoder code we cannot
   account for, and the table is checked against ggml's compiled output, not trusted). The encoder (`quantize_row_iq3_s_impl`) builds a neighbour structure at
   init; the work is the largest here.
3. Later, only on a go: `IQ2_*`, `IQ3_XXS`, `IQ1_*` (Fuel dequantizes them; no consumer has asked for them to be encoded).

## Importance vector

ggml's quantizers take an optional per-input-channel weight (the imatrix). API: `Option<&[f32]>` of length `ic`, caller supplied; we read the `.imatrix` GGUF
only if a consumer asks (a corpus file exists at `C:/Models/imatrix`). Reports measure plain (unweighted) weight error and, with a weight vector, also its weighted error.

## Error and verification

Every conversion returns an `ErrorReport` with a bound where one is provable (IQ4_NL: half the largest gap of the table times `d`, plus the f16 rounding of `d`; the grid
formats are measured only unless a bound is derived and tested). Byte-identity with ggml is the main check; dequantization is also compared with Fuel's `to_float` on random blocks.
Not verified at first: no real IQ file's blocks beyond the golden inputs.

## Not done (patent and scope)

No per-tensor format choice from measured error (H23, H4, H18): the caller names the type. No learned or per-group codebook, no lattice codebooks (idea 20 named-unread). No kernel.

## Open for the PM

1. Start with IQ4_NL + IQ4_XS in one PR or two? Recommended two (the XS scale packing is the risky part). 2. IQ3_S before or after both IQ4 types? Recommended after. 3. Version: 0.1.35 carried here, you allocate.
