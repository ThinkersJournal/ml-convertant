//! Byte-for-byte comparison against llama.cpp's `quantize_iq4_xs`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`; vectors built as for IQ4_NL (`tools/golden/iq4_xs_reference.c`,
//! `gen_iq4_xs_inputs.py`), with and without an importance vector (one 512-element row repeated over 45 rows).

use ml_convertant_core::Conversion;
use ml_convertant_ggml::{Iq4Xs, dequantize_iq4_xs, quantize_iq4_xs, quantize_iq4_xs_weighted};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

const INPUT: &[u8] = include_bytes!("golden/iq4_xs.in.f32");
const IMPORTANCE: &[u8] = include_bytes!("golden/iq4_xs.imp.f32");
const ENCODED: &[u8] = include_bytes!("golden/iq4_xs.out.bin");
const DECODED: &[u8] = include_bytes!("golden/iq4_xs.deq.f32");
const ENCODED_W: &[u8] = include_bytes!("golden/iq4_xs_w.out.bin");
const DECODED_W: &[u8] = include_bytes!("golden/iq4_xs_w.deq.f32");

fn same_bits(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

#[test]
fn encode_is_byte_identical() {
    assert_eq!(quantize_iq4_xs(&f32s(INPUT)).unwrap(), ENCODED);
}

#[test]
fn encode_with_importance_is_byte_identical() {
    let w = f32s(IMPORTANCE);
    assert_eq!(
        quantize_iq4_xs_weighted(&f32s(INPUT), Some(&w)).unwrap(),
        ENCODED_W
    );
    assert_ne!(
        ENCODED, ENCODED_W,
        "the importance vector must matter on this input"
    );
}

#[test]
fn decode_is_bit_identical() {
    same_bits(&dequantize_iq4_xs(ENCODED).unwrap(), &f32s(DECODED));
    same_bits(&dequantize_iq4_xs(ENCODED_W).unwrap(), &f32s(DECODED_W));
}

#[test]
fn the_vectors_exercise_the_format() {
    // Every table entry occurs as a nibble, every 6-bit scale code at both ends occurs (0 and 63, so both
    // clamps and the sign range), and both the high-bit and the low-nibble fields take several values.
    let mut nib = [false; 16];
    let mut codes = [false; 64];
    for block in ENCODED.as_chunks::<136>().0 {
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        for ib in 0..8 {
            let ls = usize::from((block[4 + ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (usize::from((scales_h >> (2 * ib)) & 3) << 4);
            codes[ls] = true;
        }
        for &b in &block[8..] {
            nib[usize::from(b & 15)] = true;
            nib[usize::from(b >> 4)] = true;
        }
    }
    assert_eq!(nib, [true; 16]);
    assert!(codes[0] && codes[63], "scale clamps -32 and 31 must occur");
    assert!(codes[32], "scale 0 must occur");
    assert!(codes.iter().filter(|c| **c).count() >= 40);
}

#[test]
fn report_on_golden_input() {
    let x = f32s(INPUT);
    let r = Iq4Xs::default().convert(&x).unwrap().report;
    // Only the blocks whose f16 `d` overflows decode to non-finite values.
    assert_eq!(r.nonfinite_in, 0);
    assert!(r.nonfinite_out > 0);
    assert_eq!(r.bound, Some(f64::INFINITY));
    let w = f32s(IMPORTANCE);
    let finite_rows = |n: usize| -> Vec<f32> {
        x.chunks_exact(n)
            .filter(|r| r.iter().all(|v| v.abs() < 1.0e9))
            .flatten()
            .copied()
            .collect()
    };
    for (importance, data) in [
        (None, finite_rows(256)),
        (Some(w.as_slice()), finite_rows(512)),
    ] {
        assert!(data.len() < x.len());
        let r = Iq4Xs { importance }.convert(&data).unwrap().report;
        assert_eq!(r.within_bound(), Some(true));
        assert!(r.is_clean());
    }
}
