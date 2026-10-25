//! Q4_0: 32 elements per block, 18 bytes: a little-endian IEEE half scale `d` followed by 16 bytes
//! of nibbles. Byte `j` holds element `j` in its low nibble and element `j + 16` in its high nibble;
//! a nibble `q` decodes to `(q - 8) * d`.
//!
//! Encoding follows llama.cpp's `quantize_row_q4_0_ref` in f32: `d = max / -8` where `max` is the
//! signed element of largest magnitude (the first one on ties), `id = 1 / d`, and
//! `q = min(15, trunc(x * id + 8.5))`. The scale is stored as f16 with round-to-nearest-even.
//!
//! ## Error bound
//!
//! With `d32` the f32 scale, `d16` the stored scale and `amax` the block's largest magnitude:
//!
//! ```text
//! |x - (q - 8) * d16| <= |d32| * (1 + 2^-15) + 8 * |d32 - d16| + amax * 2^-23
//! ```
//!
//! `x * id` lies in `[-8, 8]` up to rounding. Truncating `x * id + 8.5` rounds to the nearest grid
//! point (error at most 0.5) except at `+8`, where the clamp to 15 leaves error up to 1; the
//! `2^-15` term covers the f32 rounding of `id` and of the product. `|q - 8| <= 8` carries the scale
//! rounding, and `amax * 2^-23` the final f32 product. An infinite `d16` gives an infinite bound.
//!
//! Non-finite inputs and blocks with a subnormal f32 scale are errors, as for Q8_0 (the reference
//! is undefined there).

use crate::{BlockError, Q4_0Error};
use ml_convertant_core::{Conversion, Converted, ErrorReport};

/// Elements per Q4_0 block.
pub const QK4_0: usize = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 2 + QK4_0 / 2;

/// The largest magnitude and the signed element that has it (the first on ties), as the reference
/// picks it.
fn signed_max(chunk: &[f32]) -> (f32, f32) {
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in chunk {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    (amax, max)
}

/// Encodes `x` (a multiple of 32 finite values) to Q4_0 bytes.
pub fn quantize_q4_0(x: &[f32]) -> Result<Vec<u8>, Q4_0Error> {
    if !x.len().is_multiple_of(QK4_0) {
        return Err(BlockError::NotBlockAligned { len: x.len() });
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(BlockError::NonFinite { index });
    }
    let mut out = Vec::with_capacity(x.len() / QK4_0 * Q4_0_BLOCK_BYTES);
    for (block, chunk) in x.as_chunks::<QK4_0>().0.iter().enumerate() {
        let (_, max) = signed_max(chunk);
        let d = max / -8.0;
        if d != 0.0 && !(1.0f32 / d).is_finite() {
            return Err(BlockError::SubnormalScale { block });
        }
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..QK4_0 / 2 {
            let lo = ((chunk[j] * id + 8.5) as i8).min(15) as u8;
            let hi = ((chunk[j + QK4_0 / 2] * id + 8.5) as i8).min(15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    Ok(out)
}

/// Decodes Q4_0 bytes to f32.
pub fn dequantize_q4_0(bytes: &[u8]) -> Result<Vec<f32>, Q4_0Error> {
    if !bytes.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q4_0_BLOCK_BYTES,
        });
    }
    let mut out = vec![0.0f32; bytes.len() / Q4_0_BLOCK_BYTES * QK4_0];
    for (block, dst) in bytes
        .as_chunks::<Q4_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK4_0>().0)
    {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for (j, &q) in block[2..].iter().enumerate() {
            dst[j] = f32::from(i16::from(q & 0x0F) - 8) * d;
            dst[j + QK4_0 / 2] = f32::from(i16::from(q >> 4) - 8) * d;
        }
    }
    Ok(out)
}

/// The proven worst-case |x - x'| for one block of `chunk` (see the module docs).
pub fn q4_0_block_bound(chunk: &[f32]) -> f64 {
    let (amax, max) = signed_max(chunk);
    let d32 = max / -8.0;
    let d16 = half::f16::from_f32(d32).to_f32();
    if d16.is_infinite() {
        return f64::INFINITY;
    }
    let (d32, d16, amax) = (f64::from(d32), f64::from(d16), f64::from(amax));
    d32.abs() * (1.0 + 2f64.powi(-15)) + 8.0 * (d32 - d16).abs() + amax * 2f64.powi(-23)
}

/// f32 -> Q4_0, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q4_0;

impl Conversion for Q4_0 {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = Q4_0Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, Q4_0Error> {
        let output = quantize_q4_0(input)?;
        let decoded = dequantize_q4_0(&output)?;
        let bound = input
            .as_chunks::<QK4_0>()
            .0
            .iter()
            .map(|c| q4_0_block_bound(c))
            .fold(0.0, f64::max);
        let report = ErrorReport::measure(input, &decoded)
            .map_err(BlockError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_constants() {
        assert_eq!(Q4_0_BLOCK_BYTES, 18);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            quantize_q4_0(&[0.0; 31]),
            Err(BlockError::NotBlockAligned { len: 31 })
        );
        let mut v = [0.0f32; 32];
        v[3] = f32::INFINITY;
        assert_eq!(quantize_q4_0(&v), Err(BlockError::NonFinite { index: 3 }));
        let mut s = [0.0f32; 32];
        s[0] = 1.0e-40;
        assert_eq!(
            quantize_q4_0(&s),
            Err(BlockError::SubnormalScale { block: 0 })
        );
        assert_eq!(
            dequantize_q4_0(&[0; 17]),
            Err(BlockError::BadByteLength {
                len: 17,
                block_bytes: 18
            })
        );
    }

    #[test]
    fn nibble_order_and_scale_sign() {
        // max = -8 -> d = 1; element 0 -> q 0, element 16 (= 7) -> q 15.
        let mut v = [0.0f32; 32];
        v[0] = -8.0;
        v[16] = 7.0;
        let q = quantize_q4_0(&v).unwrap();
        assert_eq!(&q[..2], &half::f16::from_f32(1.0).to_bits().to_le_bytes());
        assert_eq!(q[2], 0xF0);
        assert_eq!(q[3], 0x88);
    }

    #[test]
    fn zeros_round_trip_exactly() {
        let c = Q4_0.convert(&[0.0; 64]).unwrap();
        assert_eq!(c.report.max_abs, 0.0);
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn bound_holds_on_random_blocks() {
        let mut s = 7u64;
        for round in 0..2000 {
            let scale = 10f32.powi((round % 9) - 4);
            let v: Vec<f32> = (0..64).map(|_| lcg(&mut s) * scale).collect();
            let c = Q4_0.convert(&v).unwrap();
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
        }
    }
}
