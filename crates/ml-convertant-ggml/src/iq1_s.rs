//! IQ1_S (decode only): 1.5625 bits per weight from a fixed grid of 2048 eight-element points of values in
//! `{-1, 0, 1}`. 256 elements per block, 50 bytes: an f16 `d`, `qs[32]` (the low eight bits of a grid index per eight
//! elements) and `qh[16]` (eight little-endian u16, one per 32-element sub-block: bits `3l..3l+3` are the three high
//! index bits of group `l`, bits 12..15 a 3-bit scale `s`, bit 15 the sign of the shift). A sub-block decodes to
//! `d * (2s + 1) * (g + delta)` with `g` the signed grid byte and `delta = +-0.125`.
//!
//! Table: ggml's `iq1s_grid` (`iq_tables.rs`, generated from the pinned llama.cpp by `tools/golden/gen_iq_tables.py`;
//! CI regenerates and diffs). No encoder (decode only, as the other IQ2/IQ3 decoders).

use half::f16;

use crate::BlockError;
use crate::iq_tables::IQ1S_GRID;
use crate::iq4_nl::Iq4Error;

/// Elements per IQ1_S block.
pub const QK_IQ1_S: usize = 256;
/// Bytes per IQ1_S block: 2 (f16 `d`) + 32 + 16.
pub const IQ1_S_BLOCK_BYTES: usize = 2 + QK_IQ1_S / 8 + QK_IQ1_S / 16;

/// ggml's `IQ1S_DELTA`.
pub(crate) const IQ1S_DELTA: f32 = 0.125;

/// Decodes IQ1_S bytes to f32, as ggml's `dequantize_row_iq1_s`.
pub fn dequantize_iq1_s(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ1_S_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ1_S_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ1_S_BLOCK_BYTES * QK_IQ1_S];
    for (block, dst) in bytes
        .as_chunks::<IQ1_S_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ1_S>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..34];
        for ib in 0..QK_IQ1_S / 32 {
            let qh = usize::from(u16::from_le_bytes([block[34 + 2 * ib], block[35 + 2 * ib]]));
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let idx = usize::from(qs[4 * ib + l]) | (((qh >> (3 * l)) & 7) << 8);
                let grid = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    dst[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta);
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
        assert_eq!(IQ1_S_BLOCK_BYTES, 50);
    }

    #[test]
    fn rejects_partial_block() {
        assert!(dequantize_iq1_s(&[0u8; 49]).is_err());
        assert_eq!(dequantize_iq1_s(&[]).unwrap().len(), 0);
    }
}
