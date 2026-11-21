//! IQ1_M decode against llama.cpp's `dequantize_row_iq1_m`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`. The input is 768 pseudo-random blocks
//! (`tools/golden/gen_iq_blocks.py ... iq1_m`, seed 20261120: the f16 `d` lives in the top nibbles of `scales`);
//! the expected output is ggml's, from `tools/golden/iq_dequant_reference.c iq1_m` (CI re-runs it and compares).

use ml_convertant_ggml::{IQ1_M_BLOCK_BYTES, dequantize_iq1_m};

const BLOCKS: &[u8] = include_bytes!("golden/iq1_m.blocks.bin");
const DECODED: &[u8] = include_bytes!("golden/iq1_m.deq.f32");

#[test]
fn decode_matches_ggml_bit_for_bit() {
    let want: Vec<f32> = DECODED
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let got = dequantize_iq1_m(BLOCKS).unwrap();
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), BLOCKS.len() / IQ1_M_BLOCK_BYTES * 256);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

/// Control: the vector exercises the whole 2048-point grid and all four per-group shift signs' two values.
#[test]
fn vector_covers_every_grid_point_and_both_signs() {
    let mut grid = [false; 2048];
    let mut signs = [false; 2];
    for b in BLOCKS.as_chunks::<IQ1_M_BLOCK_BYTES>().0 {
        let (qs, qh) = (&b[0..32], &b[32..48]);
        for ib in 0..8 {
            let (h0, h1) = (usize::from(qh[2 * ib]), usize::from(qh[2 * ib + 1]));
            let idx = [
                usize::from(qs[4 * ib]) | ((h0 << 8) & 0x700),
                usize::from(qs[4 * ib + 1]) | ((h0 << 4) & 0x700),
                usize::from(qs[4 * ib + 2]) | ((h1 << 8) & 0x700),
                usize::from(qs[4 * ib + 3]) | ((h1 << 4) & 0x700),
            ];
            for i in idx {
                grid[i] = true;
            }
            for bit in [h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80] {
                signs[usize::from(bit != 0)] = true;
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
