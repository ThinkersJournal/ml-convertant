//! Byte-for-byte comparison against llama.cpp's `quantize_iq4_nl`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`; vectors built as for Q8_0 (`tools/golden/iq4_nl_reference.c`,
//! `gen_iq4_nl_inputs.py`), with and without an importance vector (one 64-element row repeated over 56 rows).

use ml_convertant_core::Conversion;
use ml_convertant_ggml::{Iq4Nl, dequantize_iq4_nl, quantize_iq4_nl, quantize_iq4_nl_weighted};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

const INPUT: &[u8] = include_bytes!("golden/iq4_nl.in.f32");
const IMPORTANCE: &[u8] = include_bytes!("golden/iq4_nl.imp.f32");
const ENCODED: &[u8] = include_bytes!("golden/iq4_nl.out.bin");
const DECODED: &[u8] = include_bytes!("golden/iq4_nl.deq.f32");
const ENCODED_W: &[u8] = include_bytes!("golden/iq4_nl_w.out.bin");
const DECODED_W: &[u8] = include_bytes!("golden/iq4_nl_w.deq.f32");

fn same_bits(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

#[test]
fn encode_is_byte_identical() {
    // Includes all-zero, negative-zero and below-threshold blocks (nibbles 8) and the f16 overflow block.
    assert_eq!(quantize_iq4_nl(&f32s(INPUT)).unwrap(), ENCODED);
}

#[test]
fn the_vectors_exercise_every_table_entry() {
    // Each of the sixteen table values must occur as a nibble, so the byte-for-byte decode check covers the whole table.
    let mut seen = [false; 16];
    for block in ENCODED.as_chunks::<18>().0 {
        for &b in &block[2..] {
            seen[usize::from(b & 15)] = true;
            seen[usize::from(b >> 4)] = true;
        }
    }
    assert_eq!(seen, [true; 16]);
}

#[test]
fn encode_with_importance_is_byte_identical() {
    let w = f32s(IMPORTANCE);
    assert_eq!(
        quantize_iq4_nl_weighted(&f32s(INPUT), Some(&w)).unwrap(),
        ENCODED_W
    );
    assert_ne!(
        ENCODED, ENCODED_W,
        "the importance vector must matter on this input"
    );
}

#[test]
fn decode_is_bit_identical() {
    same_bits(&dequantize_iq4_nl(ENCODED).unwrap(), &f32s(DECODED));
    same_bits(&dequantize_iq4_nl(ENCODED_W).unwrap(), &f32s(DECODED_W));
}

#[test]
fn report_on_golden_input() {
    let x = f32s(INPUT);
    let r = Iq4Nl::default().convert(&x).unwrap().report;
    // Only the f16 scale-overflow block decodes to non-finite values.
    assert_eq!(r.nonfinite_in, 0);
    assert!(r.nonfinite_out > 0 && r.nonfinite_out <= 32);
    assert_eq!(r.bound, Some(f64::INFINITY));
    let rest: Vec<f32> = x
        .as_chunks::<32>()
        .0
        .iter()
        .filter(|b| b.iter().all(|v| v.abs() < 1.0e6))
        .flatten()
        .copied()
        .collect();
    assert!(rest.len() < x.len());
    let w = f32s(IMPORTANCE);
    // Rows of 64 elements: keep whole rows only so the importance vector still lines up.
    let rows: Vec<f32> = x
        .as_chunks::<64>()
        .0
        .iter()
        .filter(|r| r.iter().all(|v| v.abs() < 1.0e6))
        .flatten()
        .copied()
        .collect();
    for (importance, data) in [(None, &rest), (Some(w.as_slice()), &rows)] {
        let r = Iq4Nl { importance }.convert(data).unwrap().report;
        assert_eq!(r.within_bound(), Some(true));
        assert!(r.is_clean());
    }
}
