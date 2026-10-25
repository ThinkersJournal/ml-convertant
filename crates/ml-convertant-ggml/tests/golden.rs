//! Byte-for-byte comparison against llama.cpp's reference, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`. Vectors were produced by compiling that commit's
//! `ggml-quants.c` unmodified (`tools/golden/`); no llama.cpp code is in this crate.

use ml_convertant_core::Conversion;
use ml_convertant_ggml::{Q8_0, Q8_0Error, dequantize_q8_0, quantize_q8_0};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

const INPUT: &[u8] = include_bytes!("golden/q8_0.in.f32");
const ENCODED: &[u8] = include_bytes!("golden/q8_0.out.bin");
const DECODED: &[u8] = include_bytes!("golden/q8_0.deq.f32");

#[test]
fn encode_is_byte_identical() {
    let x = f32s(INPUT);
    // Includes the f16 scale-overflow block, whose infinite scale the reference stores as-is.
    assert_eq!(quantize_q8_0(&x).unwrap(), ENCODED);
}

#[test]
fn decode_is_bit_identical() {
    let got = dequantize_q8_0(ENCODED).unwrap();
    let want = f32s(DECODED);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

#[test]
fn report_on_golden_input() {
    let x = f32s(INPUT);
    let r = Q8_0.convert(&x).unwrap().report;
    // Exactly the overflow block (32 elements) decodes to NaN; nothing else changes finiteness.
    assert_eq!(r.nonfinite_in, 0);
    assert_eq!(r.nonfinite_out, 32);
    assert_eq!(r.finiteness_changed, 32);
    assert_eq!(r.bound, Some(f64::INFINITY));
    // Without that block the bound is finite and holds.
    let rest: Vec<f32> = x
        .as_chunks::<32>()
        .0
        .iter()
        .filter(|b| b.iter().all(|v| v.abs() < 1.0e29))
        .flatten()
        .copied()
        .collect();
    assert_eq!(rest.len(), x.len() - 32);
    let r = Q8_0.convert(&rest).unwrap().report;
    assert!(r.is_clean());
    assert_eq!(r.within_bound(), Some(true));
    assert!(r.bound.unwrap().is_finite());
}

#[test]
fn undefined_reference_inputs_are_errors() {
    let mut v = [1.0f32; 32];
    v[0] = f32::INFINITY;
    assert_eq!(quantize_q8_0(&v), Err(Q8_0Error::NonFinite { index: 0 }));
}
