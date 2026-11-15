//! LoRA merge into an **NF4** base: dequantize, merge in f32, requantize with the same block size.
//!
//! The base is an [`Nf4Quantized`] holding the `[out, in]` row-major weight (`n == out * in`) with the
//! first-level f32 absmax per block. NF4 blocks run over the flattened tensor, as in bitsandbytes, so unlike
//! the ggml formats no block-size multiple is required of `in_features`.
//!
//! A double-quantized base (8-bit absmax codes) is decoded to its f32 absmax by the caller first
//! (`dequantize_nested_absmax`); the output here carries plain f32 absmax and is not double-quantized again.
//!
//! # Error
//!
//! As for [`crate::merge_lora_into_quantized`]: the reference is the exact merge of the *dequantized* base
//! in f64, the report is `decode(output)` against it, and the bound is [`crate::lora_merge_bound`] plus the
//! largest per-block NF4 bound for the merged values ([`ml_convertant_nf4::nf4_block_bound`]).

use crate::{LoraAdapter, RequantError, lora_merge_bound, merge_lora, merge_lora_f64_reference};
use ml_convertant_core::{Converted, ErrorReport};
use ml_convertant_nf4::{Nf4Error, Nf4Quantized, dequantize_nf4, nf4_block_bound, quantize_nf4};

/// Merges `adapter` into the NF4 base `base`, returning the re-encoded base with the chain's error report.
pub fn merge_lora_into_nf4(
    base: &Nf4Quantized,
    adapter: &LoraAdapter,
) -> Result<Converted<Nf4Quantized>, RequantError<Nf4Error>> {
    let w = dequantize_nf4(&base.packed, &base.absmax, base.n, base.blocksize)
        .map_err(RequantError::Format)?;
    let merged = merge_lora(&w, adapter).map_err(RequantError::Lora)?;
    let output = quantize_nf4(&merged, base.blocksize).map_err(RequantError::Format)?;
    let decoded = dequantize_nf4(&output.packed, &output.absmax, output.n, output.blocksize)
        .map_err(RequantError::Format)?;
    let reference = merge_lora_f64_reference(&w, adapter).map_err(RequantError::Lora)?;
    let requant = merged
        .chunks(base.blocksize)
        .map(nf4_block_bound)
        .fold(0.0, f64::max);
    let bound = lora_merge_bound(&w, adapter).map_err(RequantError::Lora)? + requant;
    let report = ErrorReport::measure(&reference, &decoded)
        .map_err(RequantError::Measure)?
        .with_bound(bound);
    Ok(Converted { output, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn adapter(out: usize, inn: usize, rank: usize, scale: f32, seed: u64) -> LoraAdapter {
        let mut s = seed;
        LoraAdapter {
            out_features: out,
            in_features: inn,
            rank,
            a: (0..rank * inn).map(|_| lcg(&mut s)).collect(),
            b: (0..out * rank).map(|_| lcg(&mut s)).collect(),
            scale,
        }
    }

    #[test]
    fn the_chain_is_within_its_bound_across_shapes_and_block_sizes() {
        let mut seed = 5;
        for (out, inn, bs) in [(3, 64, 64), (4, 50, 64), (2, 96, 32), (5, 33, 16)] {
            seed += 1;
            let ad = adapter(out, inn, 3, 0.7, seed);
            let mut s = seed * 11;
            let w: Vec<f32> = (0..out * inn).map(|_| lcg(&mut s)).collect();
            let base = quantize_nf4(&w, bs).unwrap();
            let got = merge_lora_into_nf4(&base, &ad).unwrap();
            assert_eq!(got.output.n, out * inn);
            assert_eq!(got.output.blocksize, bs);
            assert_eq!(got.report.within_bound(), Some(true), "{out}x{inn} bs {bs}");
            assert!(got.report.max_abs > 0.0);
        }
    }

    #[test]
    fn matches_decode_merge_encode_done_by_hand() {
        let ad = adapter(3, 64, 2, 1.0, 9);
        let mut s = 3;
        let w: Vec<f32> = (0..3 * 64).map(|_| lcg(&mut s)).collect();
        let base = quantize_nf4(&w, 64).unwrap();
        let by_hand = {
            let dec = dequantize_nf4(&base.packed, &base.absmax, base.n, 64).unwrap();
            quantize_nf4(&merge_lora(&dec, &ad).unwrap(), 64).unwrap()
        };
        assert_eq!(merge_lora_into_nf4(&base, &ad).unwrap().output, by_hand);
    }

    #[test]
    fn strength_zero_is_requantization_and_a_wrong_shape_is_refused() {
        let ad = adapter(3, 64, 2, 1.0, 4);
        let mut s = 8;
        let w: Vec<f32> = (0..3 * 64).map(|_| lcg(&mut s)).collect();
        let base = quantize_nf4(&w, 64).unwrap();
        let off = merge_lora_into_nf4(&base, &ad.clone().with_strength(0.0)).unwrap();
        let dec = dequantize_nf4(&base.packed, &base.absmax, base.n, 64).unwrap();
        assert_eq!(off.output, quantize_nf4(&dec, 64).unwrap());
        let wrong = adapter(2, 64, 2, 1.0, 4);
        assert!(matches!(
            merge_lora_into_nf4(&base, &wrong),
            Err(RequantError::Lora(_))
        ));
    }
}
