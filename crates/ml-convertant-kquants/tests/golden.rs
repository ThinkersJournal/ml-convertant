//! Byte-for-byte comparison against ggml's Q4_K reference, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`, compiled unmodified (`tools/golden/`).

use ml_convertant_core::Conversion;
use ml_convertant_kquants::{Q4K, dequantize_q4_k, quantize_q4_k};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

const INPUT: &[u8] = include_bytes!("golden/q4_k.in.f32");
const ENCODED: &[u8] = include_bytes!("golden/q4_k.out.bin");
const DECODED: &[u8] = include_bytes!("golden/q4_k.deq.f32");

#[test]
fn encode_is_byte_identical() {
    let got = quantize_q4_k(&f32s(INPUT)).unwrap();
    assert_eq!(got.len(), ENCODED.len());
    let first = (0..got.len()).find(|&i| got[i] != ENCODED[i]);
    assert_eq!(
        first,
        None,
        "first differing byte (block {:?})",
        first.map(|i| i / 144)
    );
}

#[test]
fn decode_is_bit_identical() {
    let got = dequantize_q4_k(ENCODED).unwrap();
    let want = f32s(DECODED);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

#[test]
fn report_on_golden_input() {
    let r = Q4K.convert(&f32s(INPUT)).unwrap().report;
    assert!(r.is_clean());
    assert_eq!(r.within_bound(), Some(true));
    assert_eq!(r.n, 15360);
}
