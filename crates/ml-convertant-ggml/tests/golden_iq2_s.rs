//! IQ2_S decode against llama.cpp's `dequantize_row_iq2_s`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`. The input is 512 pseudo-random blocks
//! (`tools/golden/gen_iq_blocks.py`, seed 20261116); the expected output is ggml's, from
//! `tools/golden/iq_dequant_reference.c iq2_s` (CI re-runs it and compares).

use ml_convertant_ggml::{IQ2_S_BLOCK_BYTES, dequantize_iq2_s};

const BLOCKS: &[u8] = include_bytes!("golden/iq2_s.blocks.bin");
const DECODED: &[u8] = include_bytes!("golden/iq2_s.deq.f32");

#[test]
fn decode_matches_ggml_bit_for_bit() {
    let want: Vec<f32> = DECODED
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let got = dequantize_iq2_s(BLOCKS).unwrap();
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), BLOCKS.len() / IQ2_S_BLOCK_BYTES * 256);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

/// Control: the vector exercises the whole 1024-point grid, so a wrong table entry is caught.
#[test]
fn vector_covers_every_grid_point() {
    let mut grid = [false; 1024];
    for b in BLOCKS.as_chunks::<IQ2_S_BLOCK_BYTES>().0 {
        let (qs, qh) = (&b[2..34], &b[66..74]);
        for ib in 0..8 {
            for l in 0..4 {
                let idx =
                    usize::from(qs[4 * ib + l]) | ((usize::from(qh[ib]) << (8 - 2 * l)) & 0x300);
                grid[idx] = true;
            }
        }
    }
    assert!(
        grid.iter().all(|c| *c),
        "grid indices missing: {:?}",
        grid.iter().position(|c| !c)
    );
}
