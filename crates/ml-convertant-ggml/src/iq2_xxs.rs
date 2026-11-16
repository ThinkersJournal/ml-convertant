//! IQ2_XXS (decode only): 2.0625 bits per weight from a fixed grid of 256 eight-element points. 256 elements per
//! block, 66 bytes: an f16 `d` and `qs[64]`, which holds per 32-element sub-block eight bytes: four grid indices
//! (one byte each) and a little-endian `u32` with four 7-bit sign-pattern indices in bits 0..28 and a 4-bit scale
//! `s` in bits 28..32. An element decodes to `d * (0.5 + s) * 0.25 * g * sign`, with `g` the grid byte and the sign
//! from `ksigns_iq2xs[pattern]`.
//!
//! Tables: ggml's `iq2xxs_grid`, `ksigns_iq2xs`, `kmask_iq2xs` (`iq_tables.rs`, generated from the pinned llama.cpp by
//! `tools/golden/gen_iq_tables.py`; CI regenerates and diffs). No encoder (PM, 2026-11-16: dequantize only).

use half::f16;

use crate::BlockError;
use crate::iq_tables::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::iq4_nl::Iq4Error;

/// Elements per IQ2_XXS block.
pub const QK_IQ2_XXS: usize = 256;
/// Bytes per IQ2_XXS block: 2 (f16 `d`) + 64.
pub const IQ2_XXS_BLOCK_BYTES: usize = 2 + QK_IQ2_XXS / 4;

/// Decodes IQ2_XXS bytes to f32, as ggml's `dequantize_row_iq2_xxs`.
pub fn dequantize_iq2_xxs(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ2_XXS_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ2_XXS_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ2_XXS_BLOCK_BYTES * QK_IQ2_XXS];
    for (block, dst) in bytes
        .as_chunks::<IQ2_XXS_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ2_XXS>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..];
        for ib in 0..QK_IQ2_XXS / 32 {
            let q = &qs[8 * ib..8 * ib + 8];
            let aux1 = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[usize::from(q[l])].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let neg = signs & KMASK_IQ2XS[j] != 0;
                    dst[32 * ib + 8 * l + j] =
                        db * f32::from(grid[j]) * if neg { -1.0 } else { 1.0 };
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
        assert_eq!(IQ2_XXS_BLOCK_BYTES, 66);
    }

    #[test]
    fn rejects_partial_block() {
        assert!(dequantize_iq2_xxs(&[0u8; 65]).is_err());
        assert_eq!(dequantize_iq2_xxs(&[]).unwrap().len(), 0);
    }
}
