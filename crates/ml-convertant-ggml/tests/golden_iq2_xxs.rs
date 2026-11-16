//! IQ2_XXS decode against llama.cpp's `dequantize_row_iq2_xxs`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`. The input is 64 pseudo-random blocks
//! (`tools/golden/gen_iq_blocks.py`, seed 20261116); the expected output is ggml's, from
//! `tools/golden/iq_dequant_reference.c iq2_xxs` (CI re-runs it and compares).

use ml_convertant_ggml::{IQ2_XXS_BLOCK_BYTES, dequantize_iq2_xxs};

const BLOCKS: &[u8] = include_bytes!("golden/iq2_xxs.blocks.bin");
const DECODED: &[u8] = include_bytes!("golden/iq2_xxs.deq.f32");

#[test]
fn decode_matches_ggml_bit_for_bit() {
    let want: Vec<f32> = DECODED
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let got = dequantize_iq2_xxs(BLOCKS).unwrap();
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), BLOCKS.len() / IQ2_XXS_BLOCK_BYTES * 256);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

/// Control: the vector exercises the whole grid and every sign pattern, so a wrong table entry is caught.
#[test]
fn vector_covers_every_grid_point_and_sign_pattern() {
    let mut grid = [false; 256];
    let mut signs = [false; 128];
    for b in BLOCKS.as_chunks::<IQ2_XXS_BLOCK_BYTES>().0 {
        for ib in 0..8 {
            let q = &b[2 + 8 * ib..10 + 8 * ib];
            for g in &q[..4] {
                grid[usize::from(*g)] = true;
            }
            let a = u32::from_le_bytes(q[4..8].try_into().unwrap());
            for l in 0..4 {
                signs[((a >> (7 * l)) & 127) as usize] = true;
            }
        }
    }
    assert!(
        grid.iter().all(|c| *c),
        "grid indices missing: {:?}",
        grid.iter().position(|c| !c)
    );
    assert!(
        signs.iter().all(|c| *c),
        "sign patterns missing: {:?}",
        signs.iter().position(|c| !c)
    );
}
