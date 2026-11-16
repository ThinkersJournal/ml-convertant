//! IQ2_XS (decode only): 2.3125 bits per weight from a fixed grid of 512 eight-element points. 256 elements per
//! block, 74 bytes: an f16 `d`, `qs[32]` as little-endian `u16` (per eight elements: a 9-bit grid index in bits
//! 0..9 and a 7-bit sign-pattern index in bits 9..16) and `scales[8]` (a 4-bit scale per half of each 32-element
//! sub-block, low nibble for the first sixteen elements). An element decodes to `d * (0.5 + s) * 0.25 * g * sign`,
//! with `g` the grid byte and the sign from `ksigns_iq2xs[pattern]`.
//!
//! Tables: ggml's `iq2xs_grid`, `ksigns_iq2xs`, `kmask_iq2xs` (`iq_tables.rs`, generated from the pinned llama.cpp by
//! `tools/golden/gen_iq_tables.py`; CI regenerates and diffs). No encoder (PM, 2026-11-16: dequantize only).

use half::f16;

use crate::BlockError;
use crate::iq_tables::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::iq4_nl::Iq4Error;

/// Elements per IQ2_XS block.
pub const QK_IQ2_XS: usize = 256;
/// Bytes per IQ2_XS block: 2 (f16 `d`) + 64 + 8.
pub const IQ2_XS_BLOCK_BYTES: usize = 2 + QK_IQ2_XS / 4 + QK_IQ2_XS / 32;

/// Decodes IQ2_XS bytes to f32, as ggml's `dequantize_row_iq2_xs`.
pub fn dequantize_iq2_xs(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ2_XS_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ2_XS_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ2_XS_BLOCK_BYTES * QK_IQ2_XS];
    for (block, dst) in bytes
        .as_chunks::<IQ2_XS_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ2_XS>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..66];
        let scales = &block[66..74];
        for ib in 0..QK_IQ2_XS / 32 {
            let db = [
                d * (0.5 + f32::from(scales[ib] & 0xf)) * 0.25,
                d * (0.5 + f32::from(scales[ib] >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let q = u16::from_le_bytes([qs[2 * (4 * ib + l)], qs[2 * (4 * ib + l) + 1]]);
                let grid = IQ2XS_GRID[usize::from(q & 511)].to_le_bytes();
                let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
                for j in 0..8 {
                    let neg = signs & KMASK_IQ2XS[j] != 0;
                    dst[32 * ib + 8 * l + j] =
                        db[l / 2] * f32::from(grid[j]) * if neg { -1.0 } else { 1.0 };
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
        assert_eq!(IQ2_XS_BLOCK_BYTES, 74);
    }

    #[test]
    fn rejects_partial_block() {
        assert!(dequantize_iq2_xs(&[0u8; 73]).is_err());
        assert_eq!(dequantize_iq2_xs(&[]).unwrap().len(), 0);
    }
}
