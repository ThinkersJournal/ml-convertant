//! IQ3_XXS (decode only): 3.0625 bits per weight from a fixed grid of 256 four-element points. 256 elements per
//! block, 98 bytes: an f16 `d`, `qs[64]` (one grid index per four elements) and `scales_and_signs[32]` (per
//! 32-element sub-block a little-endian `u32`: four 7-bit sign-pattern indices in bits 0..28, a 4-bit scale `s` in
//! bits 28..32). An element decodes to `d * (0.5 + s) * 0.5 * g * sign`, with `g` the grid byte and the sign
//! taken from `ksigns_iq2xs[pattern]` (the table adds the parity bit, so eight sign bits come from seven).
//!
//! The tables are ggml's `iq3xxs_grid`, `ksigns_iq2xs` and `kmask_iq2xs` (`iq_tables.rs`, generated from the pinned
//! llama.cpp by `tools/golden/gen_iq_tables.py`; CI regenerates and diffs).
//!
//! No encoder: ggml's IQ3_XXS quantizer searches a grid under an importance matrix and is out of scope (PM,
//! 2026-11-16: dequantize/convert only). Every byte pattern with a finite `d` is a valid block, so decoding fails
//! only on a length that is not a whole number of blocks.

use half::f16;

use crate::BlockError;
use crate::iq_tables::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::iq4_nl::Iq4Error;

/// Elements per IQ3_XXS block.
pub const QK_IQ3_XXS: usize = 256;
/// Bytes per IQ3_XXS block: 2 (f16 `d`) + 96.
pub const IQ3_XXS_BLOCK_BYTES: usize = 2 + 3 * (QK_IQ3_XXS / 8);

/// Decodes IQ3_XXS bytes to f32, as ggml's `dequantize_row_iq3_xxs`.
pub fn dequantize_iq3_xxs(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ3_XXS_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ3_XXS_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ3_XXS_BLOCK_BYTES * QK_IQ3_XXS];
    for (block, dst) in bytes
        .as_chunks::<IQ3_XXS_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ3_XXS>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..66];
        let sas = &block[66..98];
        for ib in 0..QK_IQ3_XXS / 32 {
            let aux = u32::from_le_bytes([
                sas[4 * ib],
                sas[4 * ib + 1],
                sas[4 * ib + 2],
                sas[4 * ib + 3],
            ]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib + 2 * l])].to_le_bytes();
                let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib + 2 * l + 1])].to_le_bytes();
                for j in 0..4 {
                    let neg1 = signs & KMASK_IQ2XS[j] != 0;
                    let neg2 = signs & KMASK_IQ2XS[j + 4] != 0;
                    dst[32 * ib + 8 * l + j] =
                        db * f32::from(g1[j]) * if neg1 { -1.0 } else { 1.0 };
                    dst[32 * ib + 8 * l + j + 4] =
                        db * f32::from(g2[j]) * if neg2 { -1.0 } else { 1.0 };
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_constants() {
        assert_eq!(IQ3_XXS_BLOCK_BYTES, 98);
    }

    #[test]
    fn rejects_partial_block() {
        assert!(dequantize_iq3_xxs(&[0u8; 97]).is_err());
        assert_eq!(dequantize_iq3_xxs(&[]).unwrap().len(), 0);
    }

    #[test]
    fn zero_block_decodes_to_zero() {
        let v = dequantize_iq3_xxs(&[0u8; 98]).unwrap();
        assert_eq!(v.len(), 256);
        // d = 0 makes every value +-0; the grid's smallest entry is 4 so nothing else is possible.
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
