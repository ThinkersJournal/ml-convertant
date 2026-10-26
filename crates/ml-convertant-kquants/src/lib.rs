//! ggml K-quants with measured error. Currently [`Q4K`].
//!
//! # Q4_K
//!
//! A super-block of 256 elements is eight sub-blocks of 32. Each sub-block `j` has a 6-bit scale
//! `sc[j]` and a 6-bit minimum `m[j]`; the super-block stores two f16 factors `d` and `dmin` and the
//! twelve bytes of packed 6-bit values, then 128 bytes of nibbles (144 bytes in all). An element
//! decodes to `d * sc[j] * q - dmin * m[j]` with `q` in `0..16`; byte `l` of each 32-byte run holds
//! sub-block `2r` in its low nibble and sub-block `2r + 1` in its high nibble.
//!
//! Encoding follows ggml's `quantize_row_q4_K_ref` and `make_qkx2_quants` operation for operation in
//! f32, so the bytes are identical to ggml's; the golden tests and the CI differential check that.
//!
//! # Error bound
//!
//! There is no data-independent worst case: the 6-bit quantization of scales and minimums can leave
//! an element outside its sub-block's decoded range, where the nibble clamps. The bound recorded in
//! the [`ErrorReport`] is therefore *for this input*, computed from the encoded block as the decoder
//! reads it. For a sub-block with decoded `d1 = d * sc` and `m1 = dmin * m`, covering
//! `[-m1, 15 * d1 - m1]`, an element has error at most `d1 * (0.5 + 2^-16)` when it lies inside the
//! range and its distance to the range when outside (the nibble clamps to the nearest end), and
//! `|x + m1|` when `d1 == 0`; plus `2^-22` of the magnitudes involved for the f32 arithmetic.
//! [`q4_k_bound`] computes the maximum over the tensor.
//!
//! Non-finite inputs are errors.

use half::f16;
use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

/// Elements per super-block.
pub const QK_K: usize = 256;
/// Bytes per Q4_K super-block: 2 + 2 + 12 + 128.
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;

/// Why a K-quant conversion failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantError {
    /// Input length is not a multiple of [`QK_K`].
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
    /// Measuring the round trip failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl fmt::Display for KQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBlockAligned { len } => {
                write!(f, "{len} elements is not a multiple of {QK_K}")
            }
            Self::BadByteLength { len, block_bytes } => {
                write!(f, "{len} bytes is not a multiple of {block_bytes}")
            }
            Self::NonFinite { index } => write!(f, "element {index} is not finite"),
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for KQuantError {}

/// ggml's `nearest_int`: round to nearest even via the 1.5 * 2^23 trick, without its debug assert
/// (release builds of ggml have none, and the result for large inputs is part of the format's bytes).
fn nearest_int(f: f32) -> i32 {
    let val = f + 12_582_912.0;
    (val.to_bits() as i32 & 0x007f_ffff) - 0x0040_0000
}

/// ggml's `make_qkx2_quants`: fits `x ~ scale * L + min'` (returned as `(scale, -min')`) with levels
/// `0..=nmax`, searching `nstep` scales between `rmin` and `rmin + rdelta * nstep`.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l_out[..n].fill(0);
        return (0.0, -min);
    }
    let nmax_f = nmax as f32;
    let mut iscale = nmax_f / (max - min);
    let mut scale = 1.0f32 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.clamp(0, nmax) as u8;
        let mut diff = scale * f32::from(l_out[i]) + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax_f) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let lf = l as f32;
            let w = weights[i];
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * f32::from(laux[i]) + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l_out[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml's `get_scale_min_k4`: the 6-bit scale and minimum of sub-block `j` from the 12 packed bytes.
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn check_finite(x: &[f32]) -> Result<(), KQuantError> {
    match x.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(KQuantError::NonFinite { index }),
        None => Ok(()),
    }
}

/// Encodes `x` (a multiple of 256 finite values) to Q4_K bytes.
pub fn quantize_q4_k(x: &[f32]) -> Result<Vec<u8>, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    check_finite(x)?;
    let mut out = Vec::with_capacity(x.len() / QK_K * Q4_K_BLOCK_BYTES);
    for sb in x.as_chunks::<QK_K>().0 {
        let mut l = [0u8; QK_K];
        let mut laux = [0u8; 32];
        let mut weights = [0.0f32; 32];
        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];
        let (mut max_scale, mut max_min) = (0.0f32, 0.0f32);
        for j in 0..QK_K / 32 {
            let xs = &sb[32 * j..32 * j + 32];
            let mut sum_x2 = 0.0f32;
            for v in xs {
                sum_x2 += v * v;
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for (w, v) in weights.iter_mut().zip(xs) {
                *w = av_x + v.abs();
            }
            let (scale, min) = make_qkx2_quants(
                15,
                xs,
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut laux,
                -1.0,
                0.1,
                20,
                false,
            );
            scales[j] = scale;
            mins[j] = min;
            if scale > max_scale {
                max_scale = scale;
            }
            if min > max_min {
                max_min = min;
            }
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut packed = [0u8; 12];
        for j in 0..QK_K / 32 {
            // C stores nearest_int into a uint8_t before the MIN(63, .).
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                packed[j] = ls;
                packed[j + 4] = lm;
            } else {
                packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                packed[j - 4] |= (ls >> 4) << 6;
                packed[j] |= (lm >> 4) << 6;
            }
        }
        let d16 = f16::from_f32(max_scale / 63.0);
        let dmin16 = f16::from_f32(max_min / 63.0);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &packed);
            let d = d16.to_f32() * f32::from(sc);
            if d == 0.0 {
                continue;
            }
            let dm = dmin16.to_f32() * f32::from(m);
            for ii in 0..32 {
                let lv = nearest_int((sb[32 * j + ii] + dm) / d);
                l[32 * j + ii] = lv.clamp(0, 15) as u8;
            }
        }
        out.extend_from_slice(&d16.to_bits().to_le_bytes());
        out.extend_from_slice(&dmin16.to_bits().to_le_bytes());
        out.extend_from_slice(&packed);
        for run in l.as_chunks::<64>().0 {
            for k in 0..32 {
                out.push(run[k] | (run[k + 32] << 4));
            }
        }
    }
    Ok(out)
}

/// Decodes Q4_K bytes to f32.
pub fn dequantize_q4_k(bytes: &[u8]) -> Result<Vec<f32>, KQuantError> {
    if !bytes.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q4_K_BLOCK_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / Q4_K_BLOCK_BYTES * QK_K);
    for block in bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0 {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let min = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let scales: &[u8; 12] = block[4..16].try_into().expect("12 bytes");
        for (run, q) in block[16..].as_chunks::<32>().0.iter().enumerate() {
            let (sc, m) = get_scale_min_k4(2 * run, scales);
            let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
            let (sc, m) = get_scale_min_k4(2 * run + 1, scales);
            let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
            out.extend(q.iter().map(|&b| d1 * f32::from(b & 0xF) - m1));
            out.extend(q.iter().map(|&b| d2 * f32::from(b >> 4) - m2));
        }
    }
    Ok(out)
}

/// The worst-case |x - x'| over `x`, given its Q4_K encoding `bytes` (see the crate docs).
pub fn q4_k_bound(x: &[f32], bytes: &[u8]) -> Result<f64, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    if bytes.len() != x.len() / QK_K * Q4_K_BLOCK_BYTES {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q4_K_BLOCK_BYTES,
        });
    }
    let slack = 2f64.powi(-22);
    let mut worst = 0.0f64;
    for (sb, block) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0)
    {
        let d = f64::from(f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32());
        let dmin = f64::from(f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32());
        let scales: &[u8; 12] = block[4..16].try_into().expect("12 bytes");
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, scales);
            let (d1, m1) = (d * f64::from(sc), dmin * f64::from(m));
            let (lo, hi) = (-m1, 15.0 * d1 - m1);
            for &v in &sb[32 * j..32 * j + 32] {
                let v = f64::from(v);
                let err = if d1 == 0.0 {
                    (v + m1).abs()
                } else if v < lo {
                    lo - v
                } else if v > hi {
                    v - hi
                } else {
                    d1 * (0.5 + 2f64.powi(-16))
                };
                worst = worst.max(err + slack * (v.abs() + hi.abs() + m1.abs()));
            }
        }
    }
    Ok(worst)
}

/// f32 -> Q4_K, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q4K;

impl Conversion for Q4K {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = KQuantError;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, KQuantError> {
        let output = quantize_q4_k(input)?;
        let decoded = dequantize_q4_k(&output)?;
        let bound = q4_k_bound(input, &output)?;
        let report = ErrorReport::measure(input, &decoded)
            .map_err(KQuantError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
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

    #[test]
    fn layout_constants() {
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
    }

    #[test]
    fn nearest_int_is_round_half_even() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(7.4), 7);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            quantize_q4_k(&[0.0; 255]),
            Err(KQuantError::NotBlockAligned { len: 255 })
        );
        let mut v = [0.0f32; 256];
        v[9] = f32::NAN;
        assert_eq!(quantize_q4_k(&v), Err(KQuantError::NonFinite { index: 9 }));
        assert_eq!(
            dequantize_q4_k(&[0; 143]),
            Err(KQuantError::BadByteLength {
                len: 143,
                block_bytes: 144
            })
        );
    }

    #[test]
    fn zeros_round_trip_exactly() {
        let c = Q4K.convert(&[0.0; 512]).unwrap();
        assert_eq!(c.report.max_abs, 0.0);
        assert_eq!(c.report.within_bound(), Some(true));
    }

    #[test]
    fn scale_min_packing_round_trips() {
        // 6-bit values placed by the encoder's rule are read back by get_scale_min_k4.
        let mut packed = [0u8; 12];
        let (ls, lm): ([u8; 8], [u8; 8]) = (
            [1, 63, 20, 33, 47, 5, 62, 17],
            [60, 2, 31, 44, 9, 63, 0, 36],
        );
        for j in 0..8 {
            if j < 4 {
                packed[j] = ls[j];
                packed[j + 4] = lm[j];
            } else {
                packed[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
                packed[j - 4] |= (ls[j] >> 4) << 6;
                packed[j] |= (lm[j] >> 4) << 6;
            }
        }
        for j in 0..8 {
            assert_eq!(
                get_scale_min_k4(j, &packed),
                (ls[j], lm[j]),
                "sub-block {j}"
            );
        }
    }

    #[test]
    fn bound_holds_on_random_data() {
        let mut s = 11u64;
        for round in 0..60usize {
            let scale = 10f32.powi((round % 6) as i32 - 3);
            let offset = [0.0f32, 3.0, -3.0][round % 3] * scale;
            let v: Vec<f32> = (0..512).map(|_| lcg(&mut s) * scale + offset).collect();
            let c = Q4K.convert(&v).unwrap();
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
        }
    }
}
