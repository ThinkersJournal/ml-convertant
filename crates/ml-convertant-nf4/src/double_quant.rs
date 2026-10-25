//! Double quantization: bitsandbytes' `compress_statistics`, which quantizes the per-block `absmax`
//! values themselves to 8 bits.
//!
//! Following `quantize_4bit(compress_statistics=True)` in bitsandbytes `functional.py`: the offset is
//! the mean of all first-level absmax values, `absmax - offset` is quantized blockwise (256 values
//! per block) against the signed 8-bit *dynamic map* with the same scaling/midpoint rule as the
//! NF4 encoder, and decoding is `map[q] * nested_absmax[i / 256] + offset` ([`crate::dequantize_nested_absmax`]).
//!
//! The dynamic map ([`dynamic_map_8bit_signed`]) is generated the way bitsandbytes'
//! `create_dynamic_map(signed=True)` does, with torch's f32 `linspace`; it is checked equal to the
//! `nested_quant_map` of a real checkpoint.
//!
//! The offset is the f64 mean rounded to f32; torch reduces in f32, so the two can differ in the last
//! bit. The offset is stored with the data and decoding uses the stored value, so the decode is exact
//! either way. The offset needs all absmax values at once: quantize a whole layer's absmax together.

use crate::Nf4Error;
use ml_convertant_core::{Conversion, Converted, ErrorReport};

/// Elements per nested block in bitsandbytes.
pub const NESTED_BLOCKSIZE: usize = 256;

/// torch's f32 `linspace(start, end, steps)`: `start + step * i` below the midpoint and
/// `end - step * (steps - 1 - i)` from it, each as one fused multiply-add (as torch's vectorized CPU
/// kernel computes it; the fused form is what reproduces a real checkpoint's map).
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            }
        })
        .collect()
}

/// The signed 8-bit dynamic quantization map (bitsandbytes `create_dynamic_map(signed=True,
/// max_exponent_bits=7, total_bits=8)`), sorted ascending.
pub fn dynamic_map_8bit_signed() -> [f32; 256] {
    // 10^(i - 6), correctly rounded as Python's `10 ** (-(7 - 1) + i)` is.
    const POW10: [f64; 7] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0];
    let mut data: Vec<f64> = Vec::with_capacity(256);
    for (i, scale) in POW10.iter().enumerate() {
        let fraction_items = (1usize << i) + 1;
        let b = linspace(0.1, 1.0, fraction_items);
        for w in b.windows(2) {
            // `python_float * f32_tensor` is an f32 multiply in torch.
            let v = *scale as f32 * ((w[0] + w[1]) / 2.0);
            data.push(f64::from(v));
            data.push(-f64::from(v));
        }
    }
    data.push(0.0);
    data.push(1.0);
    data.sort_by(f64::total_cmp);
    std::array::from_fn(|i| data[i] as f32)
}

/// Double-quantized absmax values.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleQuantAbsmax {
    /// One 8-bit code per first-level absmax.
    pub codes: Vec<u8>,
    /// One f32 scale per [`NESTED_BLOCKSIZE`] codes.
    pub nested_absmax: Vec<f32>,
    /// Mean of the first-level absmax values, subtracted before quantizing.
    pub offset: f32,
}

/// Quantizes `absmax` (all first-level absmax of a tensor) against `map`, with the offset set to
/// their mean.
pub fn quantize_absmax(absmax: &[f32], map: &[f32; 256]) -> Result<DoubleQuantAbsmax, Nf4Error> {
    let offset = if absmax.is_empty() {
        0.0
    } else {
        (absmax.iter().map(|&v| f64::from(v)).sum::<f64>() / absmax.len() as f64) as f32
    };
    quantize_absmax_with_offset(absmax, map, offset)
}

/// [`quantize_absmax`] with a caller-supplied `offset`, for reproducing an existing checkpoint
/// whose offset was computed over a larger tensor.
pub fn quantize_absmax_with_offset(
    absmax: &[f32],
    map: &[f32; 256],
    offset: f32,
) -> Result<DoubleQuantAbsmax, Nf4Error> {
    if let Some(index) = absmax.iter().position(|v| !v.is_finite()) {
        return Err(Nf4Error::NonFinite { index });
    }
    if !offset.is_finite() {
        return Err(Nf4Error::NonFinite { index: 0 });
    }
    let bounds: [f32; 255] = std::array::from_fn(|i| (map[i] + map[i + 1]) / 2.0);
    let centred: Vec<f32> = absmax.iter().map(|v| v - offset).collect();
    let full = centred.len() - centred.len() % NESTED_BLOCKSIZE;
    let mut nested_absmax = Vec::with_capacity(centred.len().div_ceil(NESTED_BLOCKSIZE));
    let mut codes = Vec::with_capacity(centred.len());
    for (b, chunk) in centred.chunks(NESTED_BLOCKSIZE).enumerate() {
        let am = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        nested_absmax.push(am);
        let den = am.max(1.0e-38);
        let inv = 1.0f32 / den;
        let partial = b * NESTED_BLOCKSIZE >= full;
        for &v in chunk {
            let scaled = if partial { v / den } else { v * inv }.clamp(-1.0, 1.0);
            codes.push(bounds.partition_point(|&m| m < scaled) as u8);
        }
    }
    Ok(DoubleQuantAbsmax {
        codes,
        nested_absmax,
        offset,
    })
}

/// The proven worst-case error of the decoded absmax against the original for one nested block.
///
/// `(h + 2^-21) * nested_absmax + 2^-22 * (|max absmax in block| + |offset|)`, where `h` is half the
/// widest gap of `map` and the second term covers subtracting and re-adding the offset in f32.
pub fn absmax_block_bound(chunk: &[f32], nested_absmax: f32, offset: f32, map: &[f32; 256]) -> f64 {
    let half_gap = map
        .windows(2)
        .map(|w| (f64::from(w[1]) - f64::from(w[0])) / 2.0)
        .fold(0.0, f64::max);
    let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    (half_gap + 2f64.powi(-21)) * f64::from(nested_absmax)
        + 2f64.powi(-22) * (f64::from(amax) + f64::from(offset.abs()))
}

/// Double quantization of absmax values with the standard dynamic map, reporting the error of the
/// decoded absmax against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct DoubleQuant;

impl Conversion for DoubleQuant {
    type Input = [f32];
    type Output = DoubleQuantAbsmax;
    type Error = Nf4Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<DoubleQuantAbsmax>, Nf4Error> {
        let map = dynamic_map_8bit_signed();
        let output = quantize_absmax(input, &map)?;
        let decoded = crate::dequantize_nested_absmax(
            &output.codes,
            &map,
            &output.nested_absmax,
            NESTED_BLOCKSIZE,
            output.offset,
        )?;
        let bound = input
            .chunks(NESTED_BLOCKSIZE)
            .zip(&output.nested_absmax)
            .map(|(c, &n)| absmax_block_bound(c, n, output.offset, &map))
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
    fn map_is_sorted_with_zero_and_one() {
        let m = dynamic_map_8bit_signed();
        assert!(m.windows(2).all(|w| w[0] < w[1]));
        assert!(m.contains(&0.0));
        assert_eq!(m[255], 1.0);
        assert!(m[0] > -1.0);
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn bound_holds_on_random_absmax() {
        let mut s = 5u64;
        for round in 0..200usize {
            let n = 1 + (round * 53) % 700;
            let scale = 10f32.powi((round % 5) as i32 - 2);
            let v: Vec<f32> = (0..n).map(|_| lcg(&mut s).abs() * scale).collect();
            let c = DoubleQuant.convert(&v).unwrap();
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert_eq!(c.output.codes.len(), n);
            assert_eq!(c.output.nested_absmax.len(), n.div_ceil(NESTED_BLOCKSIZE));
        }
    }
}
