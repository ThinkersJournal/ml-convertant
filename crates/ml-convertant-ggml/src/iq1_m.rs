//! IQ1_M (decode only): 1.75 bits per weight, IQ1_S's grid with a per-eight-element sign of the shift and 3-bit scales
//! per sixteen elements. 256 elements per block, 56 bytes: `qs[32]` (low eight bits of a grid index per eight
//! elements), `qh[16]` (per eight elements a nibble: three high index bits and the shift sign, bit 3) and `scales[8]`
//! (four little-endian u16: per 32-element sub-block two 3-bit scales; the top nibbles of the four u16 also carry the
//! f16 `d`, bits 0..3 from `sc[0]`, 4..7 from `sc[1]`, 8..11 from `sc[2]`, 12..15 from `sc[3]`).
//! A group decodes to `d * (2s + 1) * (g + delta)`, `g` the signed grid byte, `delta = +-0.125`.
//!
//! Table: ggml's `iq1s_grid` (shared with IQ1_S). No encoder.

use half::f16;

use crate::BlockError;
use crate::iq_tables::IQ1S_GRID;
use crate::iq1_s::IQ1S_DELTA;
use crate::iq4_nl::Iq4Error;

/// Elements per IQ1_M block.
pub const QK_IQ1_M: usize = 256;
/// Bytes per IQ1_M block: 32 + 16 + 8.
pub const IQ1_M_BLOCK_BYTES: usize = QK_IQ1_M / 8 + QK_IQ1_M / 16 + QK_IQ1_M / 32;

/// Decodes IQ1_M bytes to f32, as ggml's `dequantize_row_iq1_m`.
pub fn dequantize_iq1_m(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ1_M_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ1_M_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ1_M_BLOCK_BYTES * QK_IQ1_M];
    for (block, dst) in bytes
        .as_chunks::<IQ1_M_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ1_M>().0)
    {
        let qs = &block[0..32];
        let qh = &block[32..48];
        let sc: [u16; 4] =
            core::array::from_fn(|k| u16::from_le_bytes([block[48 + 2 * k], block[49 + 2 * k]]));
        let d = f16::from_bits(
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000),
        )
        .to_f32();
        for ib in 0..QK_IQ1_M / 32 {
            let s = sc[ib / 2];
            let sh = 6 * (ib % 2);
            let dl = [
                d * (2 * ((s >> sh) & 7) + 1) as f32,
                d * (2 * ((s >> (sh + 3)) & 7) + 1) as f32,
            ];
            let (h0, h1) = (usize::from(qh[2 * ib]), usize::from(qh[2 * ib + 1]));
            let idx = [
                usize::from(qs[4 * ib]) | ((h0 << 8) & 0x700),
                usize::from(qs[4 * ib + 1]) | ((h0 << 4) & 0x700),
                usize::from(qs[4 * ib + 2]) | ((h1 << 8) & 0x700),
                usize::from(qs[4 * ib + 3]) | ((h1 << 4) & 0x700),
            ];
            let neg = [h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80];
            for l in 0..4 {
                let delta = if neg[l] != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for j in 0..8 {
                    dst[32 * ib + 8 * l + j] = dl[l / 2] * (f32::from(grid[j] as i8) + delta);
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
        assert_eq!(IQ1_M_BLOCK_BYTES, 56);
    }

    #[test]
    fn rejects_partial_block() {
        assert!(dequantize_iq1_m(&[0u8; 55]).is_err());
        assert_eq!(dequantize_iq1_m(&[]).unwrap().len(), 0);
    }
}
