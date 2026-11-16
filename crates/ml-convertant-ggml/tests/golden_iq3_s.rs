//! Byte-for-byte comparison against llama.cpp's `quantize_iq3_s`, ggml-org/llama.cpp
//! `9d57ce456c94d241dde672b2db9cf18879766568`; vectors built as for IQ3_S (`tools/golden/iq3_s_reference.c`,
//! `gen_iq3_s_inputs.py`), with and without an importance vector (one 512-element row repeated over 69 rows).

use ml_convertant_core::Conversion;
use ml_convertant_ggml::{Iq3S, dequantize_iq3_s, quantize_iq3_s, quantize_iq3_s_weighted};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

const INPUT: &[u8] = include_bytes!("golden/iq3_s.in.f32");
const IMPORTANCE: &[u8] = include_bytes!("golden/iq3_s.imp.f32");
const ENCODED: &[u8] = include_bytes!("golden/iq3_s.out.bin");
const DECODED: &[u8] = include_bytes!("golden/iq3_s.deq.f32");
const ENCODED_W: &[u8] = include_bytes!("golden/iq3_s_w.out.bin");
const DECODED_W: &[u8] = include_bytes!("golden/iq3_s_w.deq.f32");

fn same_bits(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}");
    }
}

#[test]
fn encode_is_byte_identical() {
    assert_eq!(quantize_iq3_s(&f32s(INPUT)).unwrap(), ENCODED);
}

#[test]
fn encode_with_importance_is_byte_identical() {
    let w = f32s(IMPORTANCE);
    assert_eq!(
        quantize_iq3_s_weighted(&f32s(INPUT), Some(&w)).unwrap(),
        ENCODED_W
    );
    assert_ne!(
        ENCODED, ENCODED_W,
        "the importance vector must matter on this input"
    );
}

#[test]
fn decode_is_bit_identical() {
    same_bits(&dequantize_iq3_s(ENCODED).unwrap(), &f32s(DECODED));
    same_bits(&dequantize_iq3_s(ENCODED_W).unwrap(), &f32s(DECODED_W));
}

#[test]
fn the_vectors_exercise_the_format() {
    // Grid indices above 255 (the `qh` bits), every 4-bit scale value, and both sign values occur, so the
    // byte-for-byte checks cover the whole block layout.
    let (mut high, mut low) = (false, false);
    let mut scales = [false; 16];
    let mut signs = 0u8;
    for block in ENCODED.as_chunks::<110>().0 {
        for ib in 0..8 {
            for k in 0..8 {
                if block[66 + ib] >> k & 1 == 1 {
                    high = true
                } else {
                    low = true
                }
            }
        }
        for &b in &block[106..110] {
            scales[usize::from(b & 15)] = true;
            scales[usize::from(b >> 4)] = true;
        }
        for &b in &block[74..106] {
            signs |= b;
        }
    }
    assert!(high && low);
    assert!(scales.iter().filter(|s| **s).count() >= 12, "{scales:?}");
    assert_eq!(signs, 255);
}

#[test]
fn report_on_golden_input() {
    let x = f32s(INPUT);
    let r = Iq3S::default().convert(&x).unwrap().report;
    // Only the blocks whose f16 `d` overflows decode to non-finite values.
    assert_eq!(r.nonfinite_in, 0);
    assert!(r.nonfinite_out > 0);
    assert_eq!(r.bound, None, "IQ3_S claims no proven bound");
    let w = f32s(IMPORTANCE);
    let finite_rows = |n: usize| -> Vec<f32> {
        x.chunks_exact(n)
            .filter(|r| r.iter().all(|v| v.abs() < 1.0e7))
            .flatten()
            .copied()
            .collect()
    };
    for (importance, data) in [
        (None, finite_rows(256)),
        (Some(w.as_slice()), finite_rows(512)),
    ] {
        assert!(data.len() < x.len());
        let r = Iq3S { importance }.convert(&data).unwrap().report;
        assert_eq!(r.bound, None);
        assert!(r.is_clean());
    }
}

#[test]
fn exactly_zero_sub_block_differs_from_ggml_by_design() {
    // Sub-block 1 is exactly zero and sub-blocks 2..8 are not. ggml 9d57ce456c skips its output-pointer advance for
    // the zero sub-block, so its later sub-blocks decode to the wrong values; we write each where it is read.
    let x = f32s(include_bytes!("golden/iq3_s_zero_sub_block.in.f32"));
    let ggml_bytes = include_bytes!("golden/iq3_s_zero_sub_block.ggml.out.bin");
    let ggml_decoded = f32s(include_bytes!("golden/iq3_s_zero_sub_block.ggml.deq.f32"));
    let ours = quantize_iq3_s(&x).unwrap();
    assert_ne!(&ours[..], &ggml_bytes[..]);
    // Same block scale and sub-block scales and qh: only the grid indices and signs are placed differently.
    assert_eq!(ours[..2], ggml_bytes[..2]);
    assert_eq!(ours[66..74], ggml_bytes[66..74]);
    assert_eq!(ours[106..], ggml_bytes[106..]);
    let our_decoded = dequantize_iq3_s(&ours).unwrap();
    let mean_abs = |d: &[f32]| {
        (64..256)
            .map(|i| f64::from((d[i] - x[i]).abs()))
            .sum::<f64>()
            / 192.0
    };
    let (ours_err, ggml_err) = (mean_abs(&our_decoded), mean_abs(&ggml_decoded));
    assert!(ours_err < 0.3, "ours {ours_err}");
    assert!(
        ggml_err > 3.0 * ours_err,
        "ggml {ggml_err} vs ours {ours_err}"
    );
    // The zero sub-block decodes to `d` (IQ3_S has no zero), never to something large.
    assert!(our_decoded[32..64].iter().all(|v| v.abs() < 0.5));
}
