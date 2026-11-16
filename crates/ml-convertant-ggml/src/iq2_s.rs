//! IQ2_S (decode only): 2.5625 bits per weight from a fixed grid of 1024 eight-element points. 256 elements per
//! block, 82 bytes: an f16 `d`, `qs[64]` (the first 32 bytes are the low eight bits of a grid index per eight
//! elements, the last 32 a sign byte per eight elements, one bit per element, set = negative), `qh[8]` (the two
//! high index bits: bits `2l..2l+2` of `qh[ib]` belong to group `l` of sub-block `ib`) and `scales[8]` (a 4-bit
//! scale per half of each 32-element sub-block, low nibble for the first sixteen elements). An element decodes
//! to `d * (0.5 + s) * 0.25 * g * sign`, with `g` the grid byte.
//!
//! Table: ggml's `iq2s_grid` and `kmask_iq2xs` (`iq_tables.rs`, generated from the pinned llama.cpp by
//! `tools/golden/gen_iq_tables.py`; CI regenerates and diffs). No encoder (PM, 2026-11-16: dequantize only).

use half::f16;

use crate::BlockError;
use crate::iq_tables::{IQ2S_GRID, KMASK_IQ2XS};
use crate::iq4_nl::Iq4Error;

/// Elements per IQ2_S block.
pub const QK_IQ2_S: usize = 256;
/// Bytes per IQ2_S block: 2 (f16 `d`) + 64 + 8 + 8.
pub const IQ2_S_BLOCK_BYTES: usize = 2 + QK_IQ2_S / 4 + 2 * (QK_IQ2_S / 32);

/// Decodes IQ2_S bytes to f32, as ggml's `dequantize_row_iq2_s`.
pub fn dequantize_iq2_s(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ2_S_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ2_S_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ2_S_BLOCK_BYTES * QK_IQ2_S];
    for (block, dst) in bytes
        .as_chunks::<IQ2_S_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ2_S>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..34];
        let signs = &block[34..66];
        let qh = &block[66..74];
        let scales = &block[74..82];
        for ib in 0..QK_IQ2_S / 32 {
            let db = [
                d * (0.5 + f32::from(scales[ib] & 0xf)) * 0.25,
                d * (0.5 + f32::from(scales[ib] >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let idx =
                    usize::from(qs[4 * ib + l]) | ((usize::from(qh[ib]) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx].to_le_bytes();
                let s = signs[4 * ib + l];
                for j in 0..8 {
                    let neg = s & KMASK_IQ2XS[j] != 0;
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
        assert_eq!(IQ2_S_BLOCK_BYTES, 82);
    }

    #[test]
    fn rejects_partial_block() {
        assert!(dequantize_iq2_s(&[0u8; 81]).is_err());
        assert_eq!(dequantize_iq2_s(&[]).unwrap().len(), 0);
    }
}
