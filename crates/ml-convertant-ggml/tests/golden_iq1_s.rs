//! IQ1_S decode against llama.cpp's `dequantize_row_iq1_s`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`. The input is 768 pseudo-random blocks
//! (`tools/golden/gen_iq_blocks.py`, seed 20261119); the expected output is ggml's, from
//! `tools/golden/iq_dequant_reference.c iq1_s` (CI re-runs it and compares).

use ml_convertant_ggml::{IQ1_S_BLOCK_BYTES, dequantize_iq1_s};

const BLOCKS: &[u8] = include_bytes!("golden/iq1_s.blocks.bin");
const DECODED: &[u8] = include_bytes!("golden/iq1_s.deq.f32");

#[test]
fn decode_matches_ggml_bit_for_bit() {
    let want: Vec<f32> = DECODED
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let got = dequantize_iq1_s(BLOCKS).unwrap();
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), BLOCKS.len() / IQ1_S_BLOCK_BYTES * 256);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

/// Control: the vector exercises the whole 2048-point grid and both shift signs, so a wrong table entry or sign is caught.
#[test]
fn vector_covers_every_grid_point_and_both_signs() {
    let mut grid = [false; 2048];
    let mut signs = [false; 2];
    for b in BLOCKS.as_chunks::<IQ1_S_BLOCK_BYTES>().0 {
        let qs = &b[2..34];
        for ib in 0..8 {
            let qh = usize::from(u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]));
            signs[usize::from(qh & 0x8000 != 0)] = true;
            for l in 0..4 {
                grid[usize::from(qs[4 * ib + l]) | (((qh >> (3 * l)) & 7) << 8)] = true;
            }
        }
    }
    assert!(
        grid.iter().all(|c| *c),
        "grid indices missing: {:?}",
        grid.iter().position(|c| !c)
    );
    assert!(signs.iter().all(|c| *c));
}
