//! LoRA merge into a **quantized** base: dequantize, merge in f32, requantize.
//!
//! The base is a tensor already encoded in one of ggml's block formats (Q8_0, Q4_0, Q2_K to Q6_K), laid out
//! `[out, in]` row-major with `in` a multiple of the format's block size (so a block never spans rows, as in a
//! GGUF weight). The adapter is as in [`crate::LoraAdapter`].
//!
//! # Error
//!
//! The reference is the *exact* merge of the **dequantized** base: `decode(base) + scale * B @ A` evaluated
//! in f64 and rounded to f32. The base's own quantization error is not attributed to this conversion: it
//! is already in the file. The report's error is `decode(output)` against that reference, and its bound is
//! the sum of two proven bounds (triangle inequality): the f32 merge's [`crate::lora_merge_bound`] and the
//! format's own bound for requantizing the merged values ([`QuantFormat::bound`]). Requantization dominates
//! in practice; merging into a 4-bit base is expected to lose more than into an 8-bit base, and the report says
//! by how much.

use crate::{LoraAdapter, LoraError, lora_merge_bound, merge_lora, merge_lora_f64_reference};
use ml_convertant_core::{Converted, ErrorReport, LengthMismatch};
use std::fmt;

/// A block-quantized tensor format that can decode, encode and bound its own error.
pub trait QuantFormat {
    /// Elements per block (or super-block).
    const BLOCK_ELEMS: usize;
    /// The format's own failure type.
    type Error: std::error::Error + 'static;

    /// Decodes encoded bytes to f32.
    fn decode(bytes: &[u8]) -> Result<Vec<f32>, Self::Error>;
    /// Encodes finite f32 values (a multiple of [`Self::BLOCK_ELEMS`]).
    fn encode(x: &[f32]) -> Result<Vec<u8>, Self::Error>;
    /// The proven worst-case `|x - decode(bytes)|` for `bytes = encode(x)`.
    fn bound(x: &[f32], bytes: &[u8]) -> Result<f64, Self::Error>;
}

macro_rules! block_bound_format {
    ($ty:ty, $elems:expr, $enc:path, $dec:path, $blk:path, $err:ty) => {
        impl QuantFormat for $ty {
            const BLOCK_ELEMS: usize = $elems;
            type Error = $err;
            fn decode(bytes: &[u8]) -> Result<Vec<f32>, Self::Error> {
                $dec(bytes)
            }
            fn encode(x: &[f32]) -> Result<Vec<u8>, Self::Error> {
                $enc(x)
            }
            fn bound(x: &[f32], _bytes: &[u8]) -> Result<f64, Self::Error> {
                Ok(x.chunks($elems).map($blk).fold(0.0, f64::max))
            }
        }
    };
}

macro_rules! whole_bound_format {
    ($ty:ty, $enc:path, $dec:path, $bound:path) => {
        impl QuantFormat for $ty {
            const BLOCK_ELEMS: usize = ml_convertant_kquants::QK_K;
            type Error = ml_convertant_kquants::KQuantError;
            fn decode(bytes: &[u8]) -> Result<Vec<f32>, Self::Error> {
                $dec(bytes)
            }
            fn encode(x: &[f32]) -> Result<Vec<u8>, Self::Error> {
                $enc(x)
            }
            fn bound(x: &[f32], bytes: &[u8]) -> Result<f64, Self::Error> {
                $bound(x, bytes)
            }
        }
    };
}

block_bound_format!(
    ml_convertant_ggml::Q8_0,
    ml_convertant_ggml::QK8_0,
    ml_convertant_ggml::quantize_q8_0,
    ml_convertant_ggml::dequantize_q8_0,
    ml_convertant_ggml::q8_0_block_bound,
    ml_convertant_ggml::BlockError
);
block_bound_format!(
    ml_convertant_ggml::Q4_0,
    ml_convertant_ggml::QK4_0,
    ml_convertant_ggml::quantize_q4_0,
    ml_convertant_ggml::dequantize_q4_0,
    ml_convertant_ggml::q4_0_block_bound,
    ml_convertant_ggml::BlockError
);
whole_bound_format!(
    ml_convertant_kquants::Q2K,
    ml_convertant_kquants::quantize_q2_k,
    ml_convertant_kquants::dequantize_q2_k,
    ml_convertant_kquants::q2_k_bound
);
whole_bound_format!(
    ml_convertant_kquants::Q3K,
    ml_convertant_kquants::quantize_q3_k,
    ml_convertant_kquants::dequantize_q3_k,
    ml_convertant_kquants::q3_k_bound
);
whole_bound_format!(
    ml_convertant_kquants::Q4K,
    ml_convertant_kquants::quantize_q4_k,
    ml_convertant_kquants::dequantize_q4_k,
    ml_convertant_kquants::q4_k_bound
);
whole_bound_format!(
    ml_convertant_kquants::Q5K,
    ml_convertant_kquants::quantize_q5_k,
    ml_convertant_kquants::dequantize_q5_k,
    ml_convertant_kquants::q5_k_bound
);
whole_bound_format!(
    ml_convertant_kquants::Q6K,
    ml_convertant_kquants::quantize_q6_k,
    ml_convertant_kquants::dequantize_q6_k,
    ml_convertant_kquants::q6_k_bound
);

/// Why a merge into a quantized base failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequantError<E> {
    /// The adapter or the merge failed (geometry, zero dimension).
    Lora(LoraError),
    /// The quantization format failed (for example a non-finite merged value).
    Format(E),
    /// `in_features` is not a multiple of the format's block size, so blocks would span rows.
    BlockSpansRows {
        /// Input features of the layer.
        in_features: usize,
        /// Elements per block of the format.
        block_elems: usize,
    },
    /// Measuring failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl<E: fmt::Display> fmt::Display for RequantError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lora(e) => e.fmt(f),
            Self::Format(e) => e.fmt(f),
            Self::BlockSpansRows {
                in_features,
                block_elems,
            } => write!(
                f,
                "in_features {in_features} is not a multiple of the block size {block_elems}"
            ),
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for RequantError<E> {}

/// Merges `adapter` into the quantized base `base` (format `F`, `[out, in]` row-major), returning the
/// requantized bytes with the chain's error report (see the module docs).
pub fn merge_lora_into_quantized<F: QuantFormat>(
    base: &[u8],
    adapter: &LoraAdapter,
) -> Result<Converted<Vec<u8>>, RequantError<F::Error>> {
    if adapter.in_features == 0 || !adapter.in_features.is_multiple_of(F::BLOCK_ELEMS) {
        return Err(RequantError::BlockSpansRows {
            in_features: adapter.in_features,
            block_elems: F::BLOCK_ELEMS,
        });
    }
    let w = F::decode(base).map_err(RequantError::Format)?;
    let merged = merge_lora(&w, adapter).map_err(RequantError::Lora)?;
    let output = F::encode(&merged).map_err(RequantError::Format)?;
    let decoded = F::decode(&output).map_err(RequantError::Format)?;
    let reference = merge_lora_f64_reference(&w, adapter).map_err(RequantError::Lora)?;
    let bound = lora_merge_bound(&w, adapter).map_err(RequantError::Lora)?
        + F::bound(&merged, &output).map_err(RequantError::Format)?;
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

    /// Quantizes random weights with `F`, merges a random adapter, and checks the proven bound and that
    /// the output decodes to the right number of finite values.
    fn check<F: QuantFormat>(inn: usize) {
        for round in 0..12usize {
            let (out, rank) = (2 + round % 3, 1 + round % 4);
            let scale = [0.25f32, 1.0, 2.0][round % 3];
            let ad = adapter(out, inn, rank, scale, 7 + round as u64);
            let mut s = 99 + round as u64;
            let mag = 10f32.powi((round % 3) as i32 - 1);
            let w: Vec<f32> = (0..out * inn).map(|_| lcg(&mut s) * mag).collect();
            let base = F::encode(&w).unwrap();
            let c = merge_lora_into_quantized::<F>(&base, &ad).unwrap();
            assert_eq!(c.report.n, out * inn);
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
            assert_eq!(F::decode(&c.output).unwrap().len(), out * inn);
        }
    }

    #[test]
    fn q8_0_and_q4_0() {
        check::<ml_convertant_ggml::Q8_0>(64);
        check::<ml_convertant_ggml::Q4_0>(64);
    }

    #[test]
    fn k_quants() {
        check::<ml_convertant_kquants::Q2K>(256);
        check::<ml_convertant_kquants::Q3K>(256);
        check::<ml_convertant_kquants::Q4K>(256);
        check::<ml_convertant_kquants::Q5K>(512);
        check::<ml_convertant_kquants::Q6K>(256);
    }

    #[test]
    fn eight_bit_loses_less_than_four_bit() {
        let (out, inn) = (4, 256);
        let ad = adapter(out, inn, 2, 1.0, 3);
        let mut s = 5u64;
        let w: Vec<f32> = (0..out * inn).map(|_| lcg(&mut s)).collect();
        let r8 = merge_lora_into_quantized::<ml_convertant_ggml::Q8_0>(
            &ml_convertant_ggml::quantize_q8_0(&w).unwrap(),
            &ad,
        )
        .unwrap()
        .report;
        let r4 = merge_lora_into_quantized::<ml_convertant_ggml::Q4_0>(
            &ml_convertant_ggml::quantize_q4_0(&w).unwrap(),
            &ad,
        )
        .unwrap()
        .report;
        assert!(r8.rmse < r4.rmse, "q8 {} vs q4 {}", r8.rmse, r4.rmse);
    }

    #[test]
    fn zero_adapter_error_is_just_requantization() {
        let (out, inn) = (2, 64);
        let mut ad = adapter(out, inn, 1, 1.0, 11);
        ad.b.fill(0.0);
        let mut s = 21u64;
        let w: Vec<f32> = (0..out * inn).map(|_| lcg(&mut s)).collect();
        let base = ml_convertant_ggml::quantize_q8_0(&w).unwrap();
        let c = merge_lora_into_quantized::<ml_convertant_ggml::Q8_0>(&base, &ad).unwrap();
        // Requantizing a Q8_0 tensor's own values moves them by at most the format's bound.
        assert_eq!(c.report.within_bound(), Some(true));
    }

    #[test]
    fn rejects_blocks_spanning_rows_and_bad_lengths() {
        let ad = adapter(2, 48, 1, 1.0, 1);
        assert_eq!(
            merge_lora_into_quantized::<ml_convertant_ggml::Q8_0>(&[0; 68], &ad).unwrap_err(),
            RequantError::BlockSpansRows {
                in_features: 48,
                block_elems: 32
            }
        );
        let ad = adapter(2, 64, 1, 1.0, 1);
        // 3 blocks of Q8_0 (102 bytes) decode to 96 values, not 2 x 64.
        let err =
            merge_lora_into_quantized::<ml_convertant_ggml::Q8_0>(&[0; 102], &ad).unwrap_err();
        assert!(matches!(
            err,
            RequantError::Lora(LoraError::BadLen { what: "W", .. })
        ));
    }
}
