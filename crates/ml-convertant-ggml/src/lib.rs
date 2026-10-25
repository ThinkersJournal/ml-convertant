//! ggml block formats with measured error: [`Q8_0`] and [`Q4_0`].
//!
//! # Q8_0
//!
//! 32 elements per block, 34 bytes: a little-endian IEEE half scale `d` followed by 32 `i8`.
//! Encoding follows llama.cpp's `quantize_row_q8_0_ref` operation for operation in f32
//! (`d = amax / 127`, `id = 1 / d`, `q = roundf(x * id)`, scale stored as f16 with
//! round-to-nearest-even), so the bytes are identical; the golden test checks that.
//!
//! ## Error bound
//!
//! With `d32` the f32 scale, `d16` the stored scale, and `amax` the block's largest magnitude, each
//! element satisfies
//!
//! ```text
//! |x - q * d16| <= d32 * (0.5 + 2^-15) + 127 * |d32 - d16| + amax * 2^-23
//! ```
//!
//! because `|x * id - q| <= 0.5` plus the f32 rounding of `id` and of the product (together under
//! `2 * 127 * 2^-24 < 2^-15`), `|q| <= 127` carries the scale rounding, and the final f32 product
//! rounds by at most `2^-24` relative. [`q8_0_block_bound`] computes it; the largest over blocks is
//! recorded as [`ErrorReport::bound`]. When `d16` is infinite (scale overflow, input above about
//! 8.3e6) the bound is infinite and the output holds NaN, which the report counts in
//! `finiteness_changed`, as the reference's output would.
//!
//! ## Inputs the reference leaves undefined
//!
//! Non-finite inputs, and blocks whose f32 scale is subnormal (so `1 / d32` overflows and the C code
//! converts an infinity to `i8`), are errors here rather than guesses.

use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

mod q4_0;
pub use q4_0::{Q4_0, Q4_0_BLOCK_BYTES, QK4_0, dequantize_q4_0, q4_0_block_bound, quantize_q4_0};

/// Elements per Q8_0 block.
pub const QK8_0: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (`i8`).
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

/// Why a block-format conversion failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    /// Input length is not a multiple of [`QK8_0`].
    NotBlockAligned {
        /// Input length in elements.
        len: usize,
    },
    /// Encoded length is not a multiple of the format's block size.
    BadByteLength {
        /// Input length in bytes.
        len: usize,
        /// Bytes per block of the format.
        block_bytes: usize,
    },
    /// Input element at `index` is NaN or infinite.
    NonFinite {
        /// Element index.
        index: usize,
    },
    /// Block `block` has a subnormal f32 scale; the reference is undefined there.
    SubnormalScale {
        /// Block index.
        block: usize,
    },
    /// Measuring the round trip failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBlockAligned { len } => {
                write!(f, "{len} elements is not a multiple of {QK8_0}")
            }
            Self::BadByteLength { len, block_bytes } => {
                write!(f, "{len} bytes is not a multiple of {block_bytes}")
            }
            Self::NonFinite { index } => write!(f, "element {index} is not finite"),
            Self::SubnormalScale { block } => {
                write!(f, "block {block} has a subnormal f32 scale")
            }
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BlockError {}

/// Error type of [`Q8_0`].
pub type Q8_0Error = BlockError;
/// Error type of [`Q4_0`].
pub type Q4_0Error = BlockError;

/// Encodes `x` (a multiple of 32 finite values) to Q8_0 bytes.
pub fn quantize_q8_0(x: &[f32]) -> Result<Vec<u8>, Q8_0Error> {
    if !x.len().is_multiple_of(QK8_0) {
        return Err(Q8_0Error::NotBlockAligned { len: x.len() });
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(Q8_0Error::NonFinite { index });
    }
    let mut out = Vec::with_capacity(x.len() / QK8_0 * Q8_0_BLOCK_BYTES);
    for (block, chunk) in x.as_chunks::<QK8_0>().0.iter().enumerate() {
        let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        if d != 0.0 && !(1.0f32 / d).is_finite() {
            return Err(Q8_0Error::SubnormalScale { block });
        }
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        out.extend(chunk.iter().map(|v| (v * id).round() as i8 as u8));
    }
    Ok(out)
}

/// Decodes Q8_0 bytes to f32.
pub fn dequantize_q8_0(bytes: &[u8]) -> Result<Vec<f32>, Q8_0Error> {
    if !bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(Q8_0Error::BadByteLength {
            len: bytes.len(),
            block_bytes: Q8_0_BLOCK_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / Q8_0_BLOCK_BYTES * QK8_0);
    for block in bytes.as_chunks::<Q8_0_BLOCK_BYTES>().0 {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        out.extend(block[2..].iter().map(|&q| f32::from(q as i8) * d));
    }
    Ok(out)
}

/// The proven worst-case |x - x'| for one block of `chunk` (see the crate docs).
pub fn q8_0_block_bound(chunk: &[f32]) -> f64 {
    let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d32 = amax / 127.0;
    let d16 = half::f16::from_f32(d32).to_f32();
    if d16.is_infinite() {
        return f64::INFINITY;
    }
    let (d32, d16, amax) = (f64::from(d32), f64::from(d16), f64::from(amax));
    d32 * (0.5 + 2f64.powi(-15)) + 127.0 * (d32 - d16).abs() + amax * 2f64.powi(-23)
}

/// f32 -> Q8_0, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q8_0;

impl Conversion for Q8_0 {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = Q8_0Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, Q8_0Error> {
        let output = quantize_q8_0(input)?;
        let decoded = dequantize_q8_0(&output)?;
        let bound = input
            .as_chunks::<QK8_0>()
            .0
            .iter()
            .map(|c| q8_0_block_bound(c))
            .fold(0.0, f64::max);
        let report = ErrorReport::measure(input, &decoded)
            .map_err(Q8_0Error::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_constants() {
        assert_eq!(Q8_0_BLOCK_BYTES, 34);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            quantize_q8_0(&[0.0; 31]),
            Err(Q8_0Error::NotBlockAligned { len: 31 })
        );
        let mut v = [0.0f32; 32];
        v[5] = f32::NAN;
        assert_eq!(quantize_q8_0(&v), Err(Q8_0Error::NonFinite { index: 5 }));
        let mut s = [0.0f32; 32];
        s[0] = 1.0e-40;
        assert_eq!(
            quantize_q8_0(&s),
            Err(Q8_0Error::SubnormalScale { block: 0 })
        );
        assert_eq!(
            dequantize_q8_0(&[0; 33]),
            Err(Q8_0Error::BadByteLength {
                len: 33,
                block_bytes: 34
            })
        );
    }

    #[test]
    fn zeros_round_trip_exactly() {
        let c = Q8_0.convert(&[0.0; 64]).unwrap();
        assert_eq!(c.report.max_abs, 0.0);
        assert_eq!(c.report.within_bound(), Some(true));
    }

    #[test]
    fn rounds_half_away_from_zero() {
        let mut v = [0.0f32; 32];
        v[..4].copy_from_slice(&[127.0, 0.5, -0.5, 2.5]);
        let q = quantize_q8_0(&v).unwrap();
        assert_eq!(&q[2..6], &[127, 1, 0xff, 3]);
    }

    /// Deterministic generator; no dependency needed.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn bound_holds_on_random_blocks() {
        let mut s = 42u64;
        for round in 0..2000 {
            let scale = 10f32.powi((round % 9) - 4);
            let v: Vec<f32> = (0..64).map(|_| lcg(&mut s) * scale).collect();
            let c = Q8_0.convert(&v).unwrap();
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
        }
    }
}
