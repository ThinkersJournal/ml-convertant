//! bitsandbytes NF4 (4-bit NormalFloat) with measured error.
//!
//! # Format
//!
//! Elements are grouped in blocks of `blocksize` (64 in practice); each block stores its absolute
//! maximum (`absmax`, f32) and each element a 4-bit index into [`NF4_CODE`]. Two indices share a
//! byte: element `2i` in the **upper** nibble, element `2i + 1` in the lower, as bitsandbytes
//! packs `(q[::2] << 4) | q[1::2]`. An element decodes to `NF4_CODE[q] * absmax`.
//!
//! # Encoding
//!
//! Follows bitsandbytes' reference `quantize_4bit` (`backends/default/ops.py`) in f32:
//! `absmax = max |x|` per block; `scaled = clamp(x * (1 / max(absmax, 1e-38)), -1, 1)` for full
//! blocks and `clamp(x / max(absmax, 1e-38), -1, 1)` for a trailing partial block; the index is the
//! number of codebook midpoints strictly below `scaled` (ties go to the lower code). An odd element
//! count is padded with a zero element in the last low nibble.
//!
//! # Error bound
//!
//! Every element lies within half the widest codebook gap of its code, in units of the block
//! `absmax`: with `h = max_i (NF4_CODE[i+1] - NF4_CODE[i]) / 2` (about 0.1519),
//!
//! ```text
//! |x - NF4_CODE[q] * absmax| <= (h + 2^-21) * absmax
//! ```
//!
//! where `2^-21` covers the f32 rounding of the reciprocal, the product, and the decode product.
//! [`nf4_block_bound`] computes it per block; the maximum is recorded as [`ErrorReport::bound`].
//! This is the error against the f32 input and excludes any later double quantization of `absmax`.
//!
//! Non-finite inputs are errors.

use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

mod double_quant;
pub use double_quant::{
    DoubleQuant, DoubleQuantAbsmax, NESTED_BLOCKSIZE, absmax_block_bound, dynamic_map_8bit_signed,
    quantize_absmax, quantize_absmax_with_offset,
};

/// The NF4 codebook (bitsandbytes `get_4bit_type("nf4")`, from `create_normal_map(offset=0.9677083,
/// use_extra_value=True)`), sorted ascending.
pub const NF4_CODE: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Why an NF4 conversion failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4Error {
    /// `blocksize` was zero.
    ZeroBlocksize,
    /// Input element at `index` is NaN or infinite.
    NonFinite {
        /// Element index.
        index: usize,
    },
    /// Buffer lengths do not agree with `n` and `blocksize`.
    Geometry {
        /// What was inconsistent.
        what: &'static str,
    },
    /// Measuring the round trip failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl fmt::Display for Nf4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlocksize => f.write_str("blocksize must be non-zero"),
            Self::NonFinite { index } => write!(f, "element {index} is not finite"),
            Self::Geometry { what } => write!(f, "inconsistent geometry: {what}"),
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Nf4Error {}

/// NF4-encoded data: packed codes and the per-block absmax.
#[derive(Debug, Clone, PartialEq)]
pub struct Nf4Quantized {
    /// Packed codes, `ceil(n / 2)` bytes, first element in the upper nibble.
    pub packed: Vec<u8>,
    /// One absmax per block, `ceil(n / blocksize)` values.
    pub absmax: Vec<f32>,
    /// Number of elements encoded.
    pub n: usize,
    /// Elements per block.
    pub blocksize: usize,
}

/// Midpoints between adjacent codes, computed in f32 as bitsandbytes does.
fn bounds() -> [f32; 15] {
    std::array::from_fn(|i| (NF4_CODE[i] + NF4_CODE[i + 1]) / 2.0)
}

/// Encodes `x` with the given `blocksize`.
pub fn quantize_nf4(x: &[f32], blocksize: usize) -> Result<Nf4Quantized, Nf4Error> {
    if blocksize == 0 {
        return Err(Nf4Error::ZeroBlocksize);
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(Nf4Error::NonFinite { index });
    }
    let bounds = bounds();
    let full = x.len() - x.len() % blocksize;
    let mut absmax = Vec::with_capacity(x.len().div_ceil(blocksize));
    let mut codes = Vec::with_capacity(x.len() + 1);
    for (b, chunk) in x.chunks(blocksize).enumerate() {
        let am = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        absmax.push(am);
        let den = am.max(1.0e-38);
        let partial = b * blocksize >= full;
        let inv = 1.0f32 / den;
        for &v in chunk {
            let scaled = if partial { v / den } else { v * inv }.clamp(-1.0, 1.0);
            codes.push(bounds.partition_point(|&m| m < scaled) as u8);
        }
    }
    if codes.len() % 2 == 1 {
        codes.push(bounds.partition_point(|&m| m < 0.0) as u8);
    }
    let packed = codes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0] << 4) | p[1])
        .collect();
    Ok(Nf4Quantized {
        packed,
        absmax,
        n: x.len(),
        blocksize,
    })
}

/// Decodes `n` elements from `packed` and `absmax`.
pub fn dequantize_nf4(
    packed: &[u8],
    absmax: &[f32],
    n: usize,
    blocksize: usize,
) -> Result<Vec<f32>, Nf4Error> {
    if blocksize == 0 {
        return Err(Nf4Error::ZeroBlocksize);
    }
    if packed.len() != n.div_ceil(2) {
        return Err(Nf4Error::Geometry {
            what: "packed length is not ceil(n / 2)",
        });
    }
    if absmax.len() != n.div_ceil(blocksize) {
        return Err(Nf4Error::Geometry {
            what: "absmax length is not ceil(n / blocksize)",
        });
    }
    Ok((0..n)
        .map(|i| {
            let byte = packed[i / 2];
            let q = if i % 2 == 0 { byte >> 4 } else { byte & 0x0F };
            NF4_CODE[usize::from(q)] * absmax[i / blocksize]
        })
        .collect())
}

/// Decodes bitsandbytes' double-quantized absmax, all in f32:
/// `map[q[i]] * nested_absmax[i / nested_blocksize] + offset`.
pub fn dequantize_nested_absmax(
    q: &[u8],
    map: &[f32; 256],
    nested_absmax: &[f32],
    nested_blocksize: usize,
    offset: f32,
) -> Result<Vec<f32>, Nf4Error> {
    if nested_blocksize == 0 {
        return Err(Nf4Error::ZeroBlocksize);
    }
    if nested_absmax.len() != q.len().div_ceil(nested_blocksize) {
        return Err(Nf4Error::Geometry {
            what: "nested_absmax length is not ceil(len / nested_blocksize)",
        });
    }
    Ok(q.iter()
        .enumerate()
        .map(|(i, &c)| map[usize::from(c)] * nested_absmax[i / nested_blocksize] + offset)
        .collect())
}

/// The proven worst-case |x - x'| for one block (see the crate docs).
pub fn nf4_block_bound(chunk: &[f32]) -> f64 {
    let half_gap = NF4_CODE
        .windows(2)
        .map(|w| (f64::from(w[1]) - f64::from(w[0])) / 2.0)
        .fold(0.0, f64::max);
    let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    (half_gap + 2f64.powi(-21)) * f64::from(amax)
}

/// f32 -> NF4 with a fixed block size, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy)]
pub struct Nf4 {
    /// Elements per block.
    pub blocksize: usize,
}

impl Conversion for Nf4 {
    type Input = [f32];
    type Output = Nf4Quantized;
    type Error = Nf4Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<Nf4Quantized>, Nf4Error> {
        let output = quantize_nf4(input, self.blocksize)?;
        let decoded = dequantize_nf4(&output.packed, &output.absmax, output.n, self.blocksize)?;
        let bound = input
            .chunks(self.blocksize)
            .map(nf4_block_bound)
            .fold(0.0, f64::max);
        let report = ErrorReport::measure(input, &decoded)
            .map_err(Nf4Error::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_is_sorted_and_spans_unit_interval() {
        assert!(NF4_CODE.windows(2).all(|w| w[0] < w[1]));
        assert_eq!((NF4_CODE[0], NF4_CODE[7], NF4_CODE[15]), (-1.0, 0.0, 1.0));
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(quantize_nf4(&[1.0], 0), Err(Nf4Error::ZeroBlocksize));
        assert_eq!(
            quantize_nf4(&[0.0, f32::NAN], 64),
            Err(Nf4Error::NonFinite { index: 1 })
        );
        assert!(dequantize_nf4(&[0; 3], &[1.0], 4, 4).is_err());
        assert!(dequantize_nf4(&[0; 2], &[1.0, 1.0], 4, 4).is_err());
    }

    #[test]
    fn extremes_hit_the_end_codes_upper_nibble_first() {
        // absmax 2 -> +2 scales to 1.0 (code 15), -2 to -1.0 (code 0).
        let q = quantize_nf4(&[2.0, -2.0, 0.0, 0.0], 4).unwrap();
        assert_eq!(q.absmax, vec![2.0]);
        assert_eq!(q.packed, vec![0xF0, 0x77]);
    }

    #[test]
    fn odd_length_pads_with_zero_code() {
        let q = quantize_nf4(&[1.0, -1.0, 1.0], 4).unwrap();
        assert_eq!(q.packed, vec![0xF0, 0xF7]);
        assert_eq!(dequantize_nf4(&q.packed, &q.absmax, 3, 4).unwrap().len(), 3);
    }

    #[test]
    fn all_zero_block_decodes_to_zero() {
        let c = Nf4 { blocksize: 64 }.convert(&[0.0; 128]).unwrap();
        assert_eq!(c.report.max_abs, 0.0);
        assert!(c.report.is_clean());
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn bound_holds_on_random_data_including_partial_blocks() {
        let mut s = 99u64;
        for round in 0..1000usize {
            let scale = 10f32.powi((round % 9) as i32 - 4);
            let n = 1 + (round * 37) % 300;
            let v: Vec<f32> = (0..n).map(|_| lcg(&mut s) * scale).collect();
            let c = Nf4 { blocksize: 64 }.convert(&v).unwrap();
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
        }
    }
}
