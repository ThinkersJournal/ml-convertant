//! Checks against real bitsandbytes output: a 16-row slice of `model.layers.1.self_attn.k_proj`
//! from `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (NF4, double-quantized) and the same rows of the
//! original bf16 weight from `Qwen/Qwen3-0.6B`. Provenance and licence:
//! `tests/fixtures/bnb_double_quant/PROVENANCE.md`.

use ml_convertant_core::Conversion;
use ml_convertant_nf4::{
    NF4_CODE, Nf4, dequantize_nested_absmax, dequantize_nf4, nf4_block_bound, quantize_nf4,
};
use mlmf_bnb::{locate_layers, read_layer};
use mlmf_safetensors::{parse_header, parse_tensors};

const SAFETENSORS: &[u8] =
    include_bytes!("fixtures/bnb_double_quant/bnb_double_quant_k_proj_16rows.safetensors");
const ORIGINAL_BF16: &[u8] =
    include_bytes!("fixtures/bnb_double_quant/orig_k_proj_16rows_bf16.bin");
const N: usize = 16 * 1024;
const BLOCKSIZE: usize = 64;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn original() -> Vec<f32> {
    ORIGINAL_BF16
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
        .collect()
}

struct Bnb {
    weight: Vec<u8>,
    absmax_codes: Vec<u8>,
    quant_map: Vec<f32>,
    nested_absmax: Vec<f32>,
    nested_map: Vec<f32>,
    nested_offset: f32,
}

fn bnb() -> Bnb {
    let header = parse_header(SAFETENSORS).unwrap();
    let (tensors, report) = parse_tensors(SAFETENSORS, &header, "fixture").unwrap();
    assert!(report.is_empty());
    let layer = &locate_layers(&tensors)[0];
    let b = read_layer(&tensors, layer).unwrap();
    Bnb {
        weight: b.weight.unwrap().into_owned(),
        absmax_codes: b.absmax.unwrap().into_owned(),
        quant_map: f32s(&b.quant_map.unwrap()),
        nested_absmax: f32s(&b.nested_absmax.unwrap()),
        nested_map: f32s(&b.nested_quant_map.unwrap()),
        nested_offset: b.state.nested_offset.unwrap() as f32,
    }
}

#[test]
fn codebook_equals_the_checkpoints_quant_map() {
    let b = bnb();
    assert_eq!(b.quant_map.len(), 16);
    for (i, (ours, theirs)) in NF4_CODE.iter().zip(&b.quant_map).enumerate() {
        assert_eq!(ours.to_bits(), theirs.to_bits(), "code {i}");
    }
}

#[test]
fn encoding_the_original_weights_reproduces_bnbs_packed_bytes() {
    let b = bnb();
    let q = quantize_nf4(&original(), BLOCKSIZE).unwrap();
    assert_eq!(q.packed.len(), b.weight.len());
    let diff: Vec<usize> = (0..q.packed.len())
        .filter(|&i| q.packed[i] != b.weight[i])
        .collect();
    assert!(
        diff.is_empty(),
        "{} of {} bytes differ, first at {:?}",
        diff.len(),
        q.packed.len(),
        diff.first()
    );
}

#[test]
fn first_level_absmax_matches_the_double_quantized_ones_to_their_precision() {
    let b = bnb();
    let q = quantize_nf4(&original(), BLOCKSIZE).unwrap();
    let map: &[f32; 256] = b.nested_map.as_slice().try_into().unwrap();
    let theirs =
        dequantize_nested_absmax(&b.absmax_codes, map, &b.nested_absmax, 256, b.nested_offset)
            .unwrap();
    assert_eq!(theirs.len(), q.absmax.len());
    // Double quantization is 8-bit over blocks of 256 absmax values: compare against the spread of
    // each nested block, not bit-for-bit.
    let worst = q
        .absmax
        .iter()
        .zip(&theirs)
        .map(|(a, t)| f64::from((a - t).abs()))
        .fold(0.0, f64::max);
    let range = q.absmax.iter().fold(0.0f32, |m, v| m.max(*v));
    assert!(
        worst < f64::from(range) * 0.02,
        "worst first-level absmax gap {worst} against max {range}"
    );
}

#[test]
fn decoding_bnbs_bytes_reproduces_the_original_to_nf4_error_with_a_clean_report() {
    let b = bnb();
    let orig = original();
    let map: &[f32; 256] = b.nested_map.as_slice().try_into().unwrap();
    let absmax =
        dequantize_nested_absmax(&b.absmax_codes, map, &b.nested_absmax, 256, b.nested_offset)
            .unwrap();
    let decoded = dequantize_nf4(&b.weight, &absmax, N, BLOCKSIZE).unwrap();
    let report = ml_convertant_core::ErrorReport::measure(&orig, &decoded).unwrap();
    // fuel's PROVENANCE.md measured relative RMS about 0.09 on this slice.
    let rms_orig = (orig.iter().map(|v| f64::from(v * v)).sum::<f64>() / N as f64).sqrt();
    let rel = report.rmse / rms_orig;
    assert!((0.08..0.10).contains(&rel), "relative RMS {rel}");
    assert!(report.is_clean());
}

#[test]
fn round_trip_report_on_the_original_weights() {
    let orig = original();
    let c = Nf4 {
        blocksize: BLOCKSIZE,
    }
    .convert(&orig)
    .unwrap();
    assert_eq!(c.report.within_bound(), Some(true));
    assert!(c.report.is_clean());
    // The single-level bound for the worst block is the format's, not a loose guess.
    let worst = orig
        .chunks(BLOCKSIZE)
        .map(nf4_block_bound)
        .fold(0.0, f64::max);
    assert_eq!(c.report.bound, Some(worst));
}
