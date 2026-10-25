//! Byte-for-byte comparison against llama.cpp's Q4_0 reference, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`; vectors built as for Q8_0 (`tools/golden/`).

use ml_convertant_core::Conversion;
use ml_convertant_ggml::{Q4_0, dequantize_q4_0, quantize_q4_0};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

const INPUT: &[u8] = include_bytes!("golden/q4_0.in.f32");
const ENCODED: &[u8] = include_bytes!("golden/q4_0.out.bin");
const DECODED: &[u8] = include_bytes!("golden/q4_0.deq.f32");

#[test]
fn encode_is_byte_identical() {
    // Includes the f16 scale-overflow block, whose infinite scale the reference stores as-is.
    assert_eq!(quantize_q4_0(&f32s(INPUT)).unwrap(), ENCODED);
}

#[test]
fn decode_is_bit_identical() {
    let got = dequantize_q4_0(ENCODED).unwrap();
    let want = f32s(DECODED);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

#[test]
fn report_on_golden_input() {
    let x = f32s(INPUT);
    let r = Q4_0.convert(&x).unwrap().report;
    // Only the scale-overflow block decodes to non-finite values.
    assert_eq!(r.nonfinite_in, 0);
    assert!(r.nonfinite_out > 0 && r.nonfinite_out <= 32);
    assert_eq!(r.nonfinite_out, r.finiteness_changed);
    assert_eq!(r.bound, Some(f64::INFINITY));
    let rest: Vec<f32> = x
        .as_chunks::<32>()
        .0
        .iter()
        .filter(|b| b.iter().all(|v| v.abs() < 1.0e29))
        .flatten()
        .copied()
        .collect();
    assert_eq!(rest.len(), x.len() - 32);
    let r = Q4_0.convert(&rest).unwrap().report;
    assert!(r.is_clean());
    assert_eq!(r.within_bound(), Some(true));
}
