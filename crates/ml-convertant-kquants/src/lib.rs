//! ggml K-quants with measured error: [`Q2K`], [`Q3K`], [`Q4K`], [`Q5K`] and [`Q6K`].
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
//! Q5_K is the same scheme with 5-bit levels (`0..=31`): 176 bytes per super-block, the fifth bit of each
//! level in a 32-byte `qh` array; its bound has `31 * d1` in place of `15 * d1`.
//!
//! # Q6_K
//!
//! Sixteen sub-blocks of 16, each with an `i8` scale; levels `-32..=31` (6 bits: 4 in `ql`, 2 in `qh`),
//! one f16 super-block scale `d`; 210 bytes. An element decodes to `d * sc[j] * (q - 32)`. Encoding follows
//! `quantize_row_q6_K_ref` and `make_qx_quants` (squared-weight scale search). The bound is per input
//! like Q4_K's, over the decoded range `[-32, 31] * d * sc[j]`.
//!
//! # Q2_K and Q3_K
//!
//! Q2_K: sixteen sub-blocks of 16 with 4-bit scale and 4-bit minimum packed in one byte, 2-bit levels,
//! f16 `d` and `dmin`; 84 bytes; decodes to `d * (sc & 15) * q - dmin * (sc >> 4)`; encoding follows
//! `quantize_row_q2_K_ref` (`make_qkx2_quants` with absolute-error fit). Q3_K: sixteen sub-blocks of 16
//! with 6-bit signed scales (12 packed bytes), 3-bit levels `-4..=3` (2 bits in `qs`, the third in `hmask`),
//! f16 `d`; 110 bytes; encoding follows `quantize_row_q3_K_ref` and `make_q3_quants` (squared-weight fit
//! with up to five refinement passes). Bounds are per input as for Q4_K.
//!
//! Non-finite inputs are errors.

use half::f16;
use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

/// Elements per super-block.
pub const QK_K: usize = 256;
/// Bytes per Q4_K super-block: 2 + 2 + 12 + 128.
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;
/// Bytes per Q5_K super-block: 2 + 2 + 12 + 32 + 128.
pub const Q5_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
/// Bytes per Q6_K super-block: 128 + 64 + 16 + 2.
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// Bytes per Q2_K super-block: 16 + 64 + 2 + 2.
pub const Q2_K_BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2;
/// Bytes per Q3_K super-block: 32 + 64 + 12 + 2.
pub const Q3_K_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;

const GROUP_MAX_EPS: f32 = 1e-15;

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

/// The part of `quantize_row_q{4,5}_K_ref` shared by both formats: per-sub-block fits, 6-bit scales and
/// minimums packed into 12 bytes, the f16 super-block factors, and the final levels `0..=nmax` computed
/// against the decoded (rounded) factors.
fn encode_scales_and_levels(
    sb: &[f32; QK_K],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> ([u8; QK_K], [u8; 12], f16, f16) {
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
            nmax,
            xs,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut laux,
            rmin,
            rdelta,
            nstep,
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
            l[32 * j + ii] = lv.clamp(0, nmax) as u8;
        }
    }
    (l, packed, d16, dmin16)
}

/// Encodes `x` (a multiple of 256 finite values) to Q4_K bytes.
pub fn quantize_q4_k(x: &[f32]) -> Result<Vec<u8>, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    check_finite(x)?;
    let mut out = Vec::with_capacity(x.len() / QK_K * Q4_K_BLOCK_BYTES);
    for sb in x.as_chunks::<QK_K>().0 {
        let (l, packed, d16, dmin16) = encode_scales_and_levels(sb, 15, -1.0, 0.1, 20);
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

/// Encodes `x` (a multiple of 256 finite values) to Q5_K bytes.
pub fn quantize_q5_k(x: &[f32]) -> Result<Vec<u8>, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    check_finite(x)?;
    let mut out = Vec::with_capacity(x.len() / QK_K * Q5_K_BLOCK_BYTES);
    for sb in x.as_chunks::<QK_K>().0 {
        let (l, packed, d16, dmin16) = encode_scales_and_levels(sb, 31, -0.5, 0.1, 15);
        out.extend_from_slice(&d16.to_bits().to_le_bytes());
        out.extend_from_slice(&dmin16.to_bits().to_le_bytes());
        out.extend_from_slice(&packed);
        let mut qh = [0u8; QK_K / 8];
        let mut ql = [0u8; QK_K / 2];
        let (mut m1, mut m2) = (1u8, 2u8);
        for (run, n) in (0..QK_K).step_by(64).enumerate() {
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[run * 32 + j] = l1 | (l2 << 4);
            }
            m1 = m1.wrapping_shl(2);
            m2 = m2.wrapping_shl(2);
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&ql);
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

/// Decodes Q5_K bytes to f32.
pub fn dequantize_q5_k(bytes: &[u8]) -> Result<Vec<f32>, KQuantError> {
    if !bytes.len().is_multiple_of(Q5_K_BLOCK_BYTES) {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q5_K_BLOCK_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / Q5_K_BLOCK_BYTES * QK_K);
    for block in bytes.as_chunks::<Q5_K_BLOCK_BYTES>().0 {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let min = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let scales: &[u8; 12] = block[4..16].try_into().expect("12 bytes");
        let qh = &block[16..16 + QK_K / 8];
        let (mut u1, mut u2) = (1u8, 2u8);
        for (run, ql) in block[16 + QK_K / 8..]
            .as_chunks::<32>()
            .0
            .iter()
            .enumerate()
        {
            let (sc, m) = get_scale_min_k4(2 * run, scales);
            let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
            let (sc, m) = get_scale_min_k4(2 * run + 1, scales);
            let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
            out.extend(
                ql.iter().zip(qh).map(|(&b, &h)| {
                    d1 * f32::from((b & 0xF) + if h & u1 != 0 { 16 } else { 0 }) - m1
                }),
            );
            out.extend(
                ql.iter().zip(qh).map(|(&b, &h)| {
                    d2 * f32::from((b >> 4) + if h & u2 != 0 { 16 } else { 0 }) - m2
                }),
            );
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
    Ok(out)
}

/// ggml's `make_qx_quants` for `rmse_type == 1` and no weights: symmetric levels `-nmax..nmax-1`
/// (stored offset by `nmax`), weighted by `x^2`, with the 19-step scale search. Returns the scale.
fn make_qx_quants(nmax: i32, x: &[f32], l_out: &mut [u8]) -> f32 {
    let n = x.len();
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let nmax_f = nmax as f32;
    let mut iscale = -nmax_f / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l_out[i] = (l + nmax) as u8;
        let w = x[i] * x[i];
        let lf = l as f32;
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax_f + 0.1f32 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for &v in x {
            let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
            let w = v * v;
            let lf = l as f32;
            sumlx += w * v * lf;
            suml2 += w * lf * lf;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
                l_out[i] = (l + nmax) as u8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Encodes `x` (a multiple of 256 finite values) to Q6_K bytes.
pub fn quantize_q6_k(x: &[f32]) -> Result<Vec<u8>, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    check_finite(x)?;
    let mut out = Vec::with_capacity(x.len() / QK_K * Q6_K_BLOCK_BYTES);
    for sb in x.as_chunks::<QK_K>().0 {
        let mut l = [0u8; QK_K];
        let mut scales = [0.0f32; QK_K / 16];
        let (mut max_scale, mut max_abs_scale) = (0.0f32, 0.0f32);
        for ib in 0..QK_K / 16 {
            let scale = make_qx_quants(
                32,
                &sb[16 * ib..16 * ib + 16],
                &mut l[16 * ib..16 * ib + 16],
            );
            scales[ib] = scale;
            if scale.abs() > max_abs_scale {
                max_abs_scale = scale.abs();
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            out.extend_from_slice(&[0u8; Q6_K_BLOCK_BYTES]);
            continue;
        }
        let iscale = -128.0f32 / max_scale;
        let d16 = f16::from_f32(1.0f32 / iscale);
        let mut sc = [0i8; QK_K / 16];
        for ib in 0..QK_K / 16 {
            sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        }
        for j in 0..QK_K / 16 {
            let d = d16.to_f32() * f32::from(sc[j]);
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lv = nearest_int(sb[16 * j + ii] / d).clamp(-32, 31);
                l[16 * j + ii] = (lv + 32) as u8;
            }
        }
        let mut ql = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 4];
        for (half, j) in (0..QK_K).step_by(128).enumerate() {
            for k in 0..32 {
                let (q1, q2) = (l[j + k] & 0xF, l[j + k + 32] & 0xF);
                let (q3, q4) = (l[j + k + 64] & 0xF, l[j + k + 96] & 0xF);
                ql[half * 64 + k] = q1 | (q3 << 4);
                ql[half * 64 + k + 32] = q2 | (q4 << 4);
                qh[half * 32 + k] = (l[j + k] >> 4)
                    | ((l[j + k + 32] >> 4) << 2)
                    | ((l[j + k + 64] >> 4) << 4)
                    | ((l[j + k + 96] >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&v| v as u8));
        out.extend_from_slice(&d16.to_bits().to_le_bytes());
    }
    Ok(out)
}

/// Decodes Q6_K bytes to f32.
pub fn dequantize_q6_k(bytes: &[u8]) -> Result<Vec<f32>, KQuantError> {
    if !bytes.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q6_K_BLOCK_BYTES,
        });
    }
    let mut out = vec![0.0f32; bytes.len() / Q6_K_BLOCK_BYTES * QK_K];
    for (block, dst) in bytes
        .as_chunks::<Q6_K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_K>().0)
    {
        let (ql, rest) = block.split_at(QK_K / 2);
        let (qh, rest) = rest.split_at(QK_K / 4);
        let (sc, dbytes) = rest.split_at(QK_K / 16);
        let d = f16::from_bits(u16::from_le_bytes([dbytes[0], dbytes[1]])).to_f32();
        for half in 0..2 {
            let (ql, qh, sc) = (&ql[half * 64..], &qh[half * 32..], &sc[half * 8..]);
            let y = &mut dst[half * 128..half * 128 + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * f32::from(sc[is] as i8) * q1 as f32;
                y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
                y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
                y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// The worst-case |x - x'| over `x`, given its Q6_K encoding `bytes` (see the crate docs).
pub fn q6_k_bound(x: &[f32], bytes: &[u8]) -> Result<f64, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    if bytes.len() != x.len() / QK_K * Q6_K_BLOCK_BYTES {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q6_K_BLOCK_BYTES,
        });
    }
    let slack = 2f64.powi(-22);
    let mut worst = 0.0f64;
    for (sb, block) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(bytes.as_chunks::<Q6_K_BLOCK_BYTES>().0)
    {
        let sc = &block[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
        let d = f64::from(
            f16::from_bits(u16::from_le_bytes([
                block[Q6_K_BLOCK_BYTES - 2],
                block[Q6_K_BLOCK_BYTES - 1],
            ]))
            .to_f32(),
        );
        for j in 0..QK_K / 16 {
            let ds = d * f64::from(sc[j] as i8);
            let (a, b) = (-32.0 * ds, 31.0 * ds);
            let (lo, hi) = (a.min(b), a.max(b));
            for &v in &sb[16 * j..16 * j + 16] {
                let v = f64::from(v);
                let err = if ds == 0.0 {
                    v.abs()
                } else if v < lo {
                    lo - v
                } else if v > hi {
                    v - hi
                } else {
                    ds.abs() * (0.5 + 2f64.powi(-16))
                };
                worst = worst.max(err + slack * (v.abs() + lo.abs() + hi.abs()));
            }
        }
    }
    Ok(worst)
}

/// f32 -> Q6_K, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q6K;

impl Conversion for Q6K {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = KQuantError;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, KQuantError> {
        let output = quantize_q6_k(input)?;
        let decoded = dequantize_q6_k(&output)?;
        let bound = q6_k_bound(input, &output)?;
        let report = ErrorReport::measure(input, &decoded)
            .map_err(KQuantError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

/// Encodes `x` (a multiple of 256 finite values) to Q2_K bytes.
pub fn quantize_q2_k(x: &[f32]) -> Result<Vec<u8>, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    check_finite(x)?;
    let q4scale = 15.0f32;
    let mut out = Vec::with_capacity(x.len() / QK_K * Q2_K_BLOCK_BYTES);
    for sb in x.as_chunks::<QK_K>().0 {
        let mut l = [0u8; QK_K];
        let mut laux = [0u8; 16];
        let mut weights = [0.0f32; 16];
        let mut mins = [0.0f32; QK_K / 16];
        let mut scales = [0.0f32; QK_K / 16];
        let (mut max_scale, mut max_min) = (0.0f32, 0.0f32);
        for j in 0..QK_K / 16 {
            let xs = &sb[16 * j..16 * j + 16];
            for (w, v) in weights.iter_mut().zip(xs) {
                *w = v.abs();
            }
            let (scale, min) = make_qkx2_quants(
                3,
                xs,
                &weights,
                &mut l[16 * j..16 * j + 16],
                &mut laux,
                -0.5,
                0.1,
                15,
                true,
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
        let mut packed = [0u8; QK_K / 16];
        let (d16, dmin16);
        if max_scale > 0.0 {
            let iscale = q4scale / max_scale;
            for j in 0..QK_K / 16 {
                packed[j] = nearest_int(iscale * scales[j]) as u8;
            }
            d16 = f16::from_f32(max_scale / q4scale);
        } else {
            d16 = f16::from_f32(0.0);
        }
        if max_min > 0.0 {
            let iscale = q4scale / max_min;
            for j in 0..QK_K / 16 {
                let lv = nearest_int(iscale * mins[j]);
                packed[j] |= (lv << 4) as u8;
            }
            dmin16 = f16::from_f32(max_min / q4scale);
        } else {
            dmin16 = f16::from_f32(0.0);
        }
        for j in 0..QK_K / 16 {
            let d = d16.to_f32() * f32::from(packed[j] & 0xF);
            if d == 0.0 {
                continue;
            }
            let dm = dmin16.to_f32() * f32::from(packed[j] >> 4);
            for ii in 0..16 {
                let lv = nearest_int((sb[16 * j + ii] + dm) / d);
                l[16 * j + ii] = lv.clamp(0, 3) as u8;
            }
        }
        out.extend_from_slice(&packed);
        for j in (0..QK_K).step_by(128) {
            for k in 0..32 {
                out.push(
                    l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6),
                );
            }
        }
        out.extend_from_slice(&d16.to_bits().to_le_bytes());
        out.extend_from_slice(&dmin16.to_bits().to_le_bytes());
    }
    Ok(out)
}

/// Decodes Q2_K bytes to f32.
pub fn dequantize_q2_k(bytes: &[u8]) -> Result<Vec<f32>, KQuantError> {
    if !bytes.len().is_multiple_of(Q2_K_BLOCK_BYTES) {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q2_K_BLOCK_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / Q2_K_BLOCK_BYTES * QK_K);
    for block in bytes.as_chunks::<Q2_K_BLOCK_BYTES>().0 {
        let scales = &block[..QK_K / 16];
        let qs = &block[QK_K / 16..QK_K / 16 + QK_K / 4];
        let d = f16::from_bits(u16::from_le_bytes([
            block[Q2_K_BLOCK_BYTES - 4],
            block[Q2_K_BLOCK_BYTES - 3],
        ]))
        .to_f32();
        let min = f16::from_bits(u16::from_le_bytes([
            block[Q2_K_BLOCK_BYTES - 2],
            block[Q2_K_BLOCK_BYTES - 1],
        ]))
        .to_f32();
        let mut is = 0;
        for q in qs.as_chunks::<32>().0 {
            for shift in [0u32, 2, 4, 6] {
                for half in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let (dl, ml) = (d * f32::from(sc & 0xF), min * f32::from(sc >> 4));
                    out.extend(
                        q[half * 16..half * 16 + 16]
                            .iter()
                            .map(|&b| dl * f32::from((b >> shift) & 3) - ml),
                    );
                }
            }
        }
    }
    Ok(out)
}

/// The worst-case |x - x'| over `x`, given its Q2_K encoding `bytes` (see the crate docs).
pub fn q2_k_bound(x: &[f32], bytes: &[u8]) -> Result<f64, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    if bytes.len() != x.len() / QK_K * Q2_K_BLOCK_BYTES {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q2_K_BLOCK_BYTES,
        });
    }
    let slack = 2f64.powi(-22);
    let mut worst = 0.0f64;
    for (sb, block) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(bytes.as_chunks::<Q2_K_BLOCK_BYTES>().0)
    {
        let h = |i: usize| {
            f64::from(f16::from_bits(u16::from_le_bytes([block[i], block[i + 1]])).to_f32())
        };
        let (d, dmin) = (h(Q2_K_BLOCK_BYTES - 4), h(Q2_K_BLOCK_BYTES - 2));
        for j in 0..QK_K / 16 {
            let (dl, ml) = (
                d * f64::from(block[j] & 0xF),
                dmin * f64::from(block[j] >> 4),
            );
            let (lo, hi) = (-ml, 3.0 * dl - ml);
            for &v in &sb[16 * j..16 * j + 16] {
                let v = f64::from(v);
                let err = if dl == 0.0 {
                    (v + ml).abs()
                } else if v < lo {
                    lo - v
                } else if v > hi {
                    v - hi
                } else {
                    dl * (0.5 + 2f64.powi(-16))
                };
                worst = worst.max(err + slack * (v.abs() + hi.abs() + ml.abs()));
            }
        }
    }
    Ok(worst)
}

/// f32 -> Q2_K, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q2K;

impl Conversion for Q2K {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = KQuantError;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, KQuantError> {
        let output = quantize_q2_k(input)?;
        let decoded = dequantize_q2_k(&output)?;
        let bound = q2_k_bound(input, &output)?;
        let report = ErrorReport::measure(input, &decoded)
            .map_err(KQuantError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

/// ggml's `make_q3_quants` with `do_rmse`: levels `-nmax..nmax-1` offset by `nmax` into `0..2*nmax`,
/// refined by up to five passes of single-element changes.
fn make_q3_quants_rmse(nmax: i32, x: &[f32], l_out: &mut [i8]) -> f32 {
    let n = x.len();
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l_out[..n].fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l_out[i] = l as i8;
        let w = x[i] * x[i];
        let lf = l as f32;
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let li = f32::from(l_out[i]);
            let mut slx = sumlx - w * x[i] * li;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * li * li;
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l != i32::from(l_out[i]) {
                    let nl = new_l as f32;
                    slx += w * x[i] * nl;
                    sl2 += w * nl * nl;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l_out[i] = new_l as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    for l in l_out[..n].iter_mut() {
        *l += nmax as i8;
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// The six-bit scale of Q3_K sub-block `j`, minus 32, from the 12 packed bytes.
fn q3_scale(scales: &[u8], j: usize) -> i32 {
    let low = if j < 8 {
        scales[j] & 0xF
    } else {
        scales[j - 8] >> 4
    };
    let high = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    i32::from(low | (high << 4)) - 32
}

/// Encodes `x` (a multiple of 256 finite values) to Q3_K bytes.
pub fn quantize_q3_k(x: &[f32]) -> Result<Vec<u8>, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    check_finite(x)?;
    let mut out = Vec::with_capacity(x.len() / QK_K * Q3_K_BLOCK_BYTES);
    for sb in x.as_chunks::<QK_K>().0 {
        let mut l = [0i8; QK_K];
        let mut scales = [0.0f32; QK_K / 16];
        let (mut max_scale, mut amax) = (0.0f32, 0.0f32);
        for j in 0..QK_K / 16 {
            scales[j] =
                make_q3_quants_rmse(4, &sb[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16]);
            let scale = scales[j].abs();
            if scale > amax {
                amax = scale;
                max_scale = scales[j];
            }
        }
        let mut packed = [0u8; 12];
        let d16 = if max_scale != 0.0 {
            let iscale = -32.0f32 / max_scale;
            for j in 0..QK_K / 16 {
                let lv = (nearest_int(iscale * scales[j]) as i8).clamp(-32, 31) + 32;
                let lv = lv as u8;
                if j < 8 {
                    packed[j] = lv & 0xF;
                } else {
                    packed[j - 8] |= (lv & 0xF) << 4;
                }
                packed[j % 4 + 8] |= (lv >> 4) << (2 * (j / 4));
            }
            f16::from_f32(1.0f32 / iscale)
        } else {
            f16::from_f32(0.0)
        };
        for j in 0..QK_K / 16 {
            let d = d16.to_f32() * q3_scale(&packed, j) as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let lv = nearest_int(sb[16 * j + ii] / d).clamp(-4, 3);
                l[16 * j + ii] = (lv + 4) as i8;
            }
        }
        let mut hmask = [0u8; QK_K / 8];
        let (mut m, mut hm) = (0usize, 1u8);
        for lj in l.iter_mut() {
            if *lj > 3 {
                hmask[m] |= hm;
                *lj -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm = hm.wrapping_shl(1);
            }
        }
        out.extend_from_slice(&hmask);
        for j in (0..QK_K).step_by(128) {
            for k in 0..32 {
                out.push(
                    (l[j + k] | (l[j + k + 32] << 2) | (l[j + k + 64] << 4) | (l[j + k + 96] << 6))
                        as u8,
                );
            }
        }
        out.extend_from_slice(&packed);
        out.extend_from_slice(&d16.to_bits().to_le_bytes());
    }
    Ok(out)
}

/// Decodes Q3_K bytes to f32.
pub fn dequantize_q3_k(bytes: &[u8]) -> Result<Vec<f32>, KQuantError> {
    if !bytes.len().is_multiple_of(Q3_K_BLOCK_BYTES) {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q3_K_BLOCK_BYTES,
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / Q3_K_BLOCK_BYTES * QK_K);
    for block in bytes.as_chunks::<Q3_K_BLOCK_BYTES>().0 {
        let hm = &block[..QK_K / 8];
        let qs = &block[QK_K / 8..QK_K / 8 + QK_K / 4];
        let scales = &block[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12];
        let d_all = f16::from_bits(u16::from_le_bytes([
            block[Q3_K_BLOCK_BYTES - 2],
            block[Q3_K_BLOCK_BYTES - 1],
        ]))
        .to_f32();
        let mut m = 1u8;
        let mut is = 0;
        for q in qs.as_chunks::<32>().0 {
            for shift in [0u32, 2, 4, 6] {
                for half in 0..2 {
                    let dl = d_all * q3_scale(scales, is) as f32;
                    is += 1;
                    out.extend((0..16).map(|l| {
                        let i = half * 16 + l;
                        let low = i32::from((q[i] >> shift) & 3);
                        dl * (low - if hm[i] & m != 0 { 0 } else { 4 }) as f32
                    }));
                }
                m = m.wrapping_shl(1);
            }
        }
    }
    Ok(out)
}

/// The worst-case |x - x'| over `x`, given its Q3_K encoding `bytes` (see the crate docs).
pub fn q3_k_bound(x: &[f32], bytes: &[u8]) -> Result<f64, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    if bytes.len() != x.len() / QK_K * Q3_K_BLOCK_BYTES {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes: Q3_K_BLOCK_BYTES,
        });
    }
    let slack = 2f64.powi(-22);
    let mut worst = 0.0f64;
    for (sb, block) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(bytes.as_chunks::<Q3_K_BLOCK_BYTES>().0)
    {
        let scales = &block[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12];
        let d = f64::from(
            f16::from_bits(u16::from_le_bytes([
                block[Q3_K_BLOCK_BYTES - 2],
                block[Q3_K_BLOCK_BYTES - 1],
            ]))
            .to_f32(),
        );
        for j in 0..QK_K / 16 {
            let ds = d * f64::from(q3_scale(scales, j));
            let (a, b) = (-4.0 * ds, 3.0 * ds);
            let (lo, hi) = (a.min(b), a.max(b));
            for &v in &sb[16 * j..16 * j + 16] {
                let v = f64::from(v);
                let err = if ds == 0.0 {
                    v.abs()
                } else if v < lo {
                    lo - v
                } else if v > hi {
                    v - hi
                } else {
                    ds.abs() * (0.5 + 2f64.powi(-16))
                };
                worst = worst.max(err + slack * (v.abs() + lo.abs() + hi.abs()));
            }
        }
    }
    Ok(worst)
}

/// f32 -> Q3_K, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q3K;

impl Conversion for Q3K {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = KQuantError;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, KQuantError> {
        let output = quantize_q3_k(input)?;
        let decoded = dequantize_q3_k(&output)?;
        let bound = q3_k_bound(input, &output)?;
        let report = ErrorReport::measure(input, &decoded)
            .map_err(KQuantError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

fn k4_bound(x: &[f32], bytes: &[u8], block_bytes: usize, nmax: f64) -> Result<f64, KQuantError> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(KQuantError::NotBlockAligned { len: x.len() });
    }
    if bytes.len() != x.len() / QK_K * block_bytes {
        return Err(KQuantError::BadByteLength {
            len: bytes.len(),
            block_bytes,
        });
    }
    let slack = 2f64.powi(-22);
    let mut worst = 0.0f64;
    for (sb, block) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(bytes.chunks_exact(block_bytes))
    {
        let d = f64::from(f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32());
        let dmin = f64::from(f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32());
        let scales: &[u8; 12] = block[4..16].try_into().expect("12 bytes");
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, scales);
            let (d1, m1) = (d * f64::from(sc), dmin * f64::from(m));
            let (lo, hi) = (-m1, nmax * d1 - m1);
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

/// The worst-case |x - x'| over `x`, given its Q4_K encoding `bytes` (see the crate docs).
pub fn q4_k_bound(x: &[f32], bytes: &[u8]) -> Result<f64, KQuantError> {
    k4_bound(x, bytes, Q4_K_BLOCK_BYTES, 15.0)
}

/// The worst-case |x - x'| over `x`, given its Q5_K encoding `bytes` (see the crate docs).
pub fn q5_k_bound(x: &[f32], bytes: &[u8]) -> Result<f64, KQuantError> {
    k4_bound(x, bytes, Q5_K_BLOCK_BYTES, 31.0)
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

/// f32 -> Q5_K, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Q5K;

impl Conversion for Q5K {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = KQuantError;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, KQuantError> {
        let output = quantize_q5_k(input)?;
        let decoded = dequantize_q5_k(&output)?;
        let bound = q5_k_bound(input, &output)?;
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

    #[test]
    fn q5_layout_and_round_trip() {
        assert_eq!(Q5_K_BLOCK_BYTES, 176);
        let mut st = 21u64;
        for round in 0..40usize {
            let scale = 10f32.powi((round % 6) as i32 - 3);
            let v: Vec<f32> = (0..512).map(|_| lcg(&mut st) * scale).collect();
            let c = Q5K.convert(&v).unwrap();
            assert_eq!(c.output.len(), 2 * Q5_K_BLOCK_BYTES);
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
            // Q5_K is at least as accurate as Q4_K on the same data (finer levels), not a bound.
            let c4 = Q4K.convert(&v).unwrap();
            assert!(c.report.rmse <= c4.report.rmse * 1.0001, "round {round}");
        }
    }

    #[test]
    fn q5_rejects_bad_input() {
        assert_eq!(
            dequantize_q5_k(&[0; 175]),
            Err(KQuantError::BadByteLength {
                len: 175,
                block_bytes: 176
            })
        );
        assert_eq!(
            quantize_q5_k(&[0.0; 100]),
            Err(KQuantError::NotBlockAligned { len: 100 })
        );
    }

    #[test]
    fn q6_layout_round_trip_and_zero_block() {
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
        let z = Q6K.convert(&[0.0; 256]).unwrap();
        assert_eq!(z.output, vec![0u8; 210]);
        assert_eq!(z.report.max_abs, 0.0);
        let mut st = 33u64;
        for round in 0..40usize {
            let scale = 10f32.powi((round % 6) as i32 - 3);
            let v: Vec<f32> = (0..512).map(|_| lcg(&mut st) * scale).collect();
            let c = Q6K.convert(&v).unwrap();
            assert_eq!(c.output.len(), 2 * Q6_K_BLOCK_BYTES);
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
        }
    }

    #[test]
    fn q6_rejects_bad_input() {
        assert_eq!(
            dequantize_q6_k(&[0; 209]),
            Err(KQuantError::BadByteLength {
                len: 209,
                block_bytes: 210
            })
        );
        assert_eq!(
            quantize_q6_k(&[0.0; 3]),
            Err(KQuantError::NotBlockAligned { len: 3 })
        );
    }

    #[test]
    fn q2_q3_layouts_and_round_trip() {
        assert_eq!(Q2_K_BLOCK_BYTES, 84);
        assert_eq!(Q3_K_BLOCK_BYTES, 110);
        let mut st = 77u64;
        for round in 0..40usize {
            let scale = 10f32.powi((round % 6) as i32 - 3);
            let offset = [0.0f32, 2.0, -2.0][round % 3] * scale;
            let v: Vec<f32> = (0..512).map(|_| lcg(&mut st) * scale + offset).collect();
            let c2 = Q2K.convert(&v).unwrap();
            assert_eq!(c2.output.len(), 2 * Q2_K_BLOCK_BYTES);
            assert_eq!(c2.report.within_bound(), Some(true), "q2 round {round}");
            assert!(c2.report.is_clean());
            let c3 = Q3K.convert(&v).unwrap();
            assert_eq!(c3.output.len(), 2 * Q3_K_BLOCK_BYTES);
            assert_eq!(c3.report.within_bound(), Some(true), "q3 round {round}");
            assert!(c3.report.is_clean());
        }
        assert_eq!(Q2K.convert(&[0.0; 256]).unwrap().report.max_abs, 0.0);
        assert_eq!(Q3K.convert(&[0.0; 256]).unwrap().report.max_abs, 0.0);
    }

    #[test]
    fn q3_scale_packing_round_trips() {
        // Encoder rule for 6-bit values placed in the 12 bytes, read back by q3_scale.
        let vals: [u8; 16] = [0, 63, 20, 33, 47, 5, 62, 17, 9, 31, 32, 1, 40, 2, 55, 60];
        let mut packed = [0u8; 12];
        for (j, &lv) in vals.iter().enumerate() {
            if j < 8 {
                packed[j] = lv & 0xF;
            } else {
                packed[j - 8] |= (lv & 0xF) << 4;
            }
            packed[j % 4 + 8] |= (lv >> 4) << (2 * (j / 4));
        }
        for (j, &lv) in vals.iter().enumerate() {
            assert_eq!(q3_scale(&packed, j), i32::from(lv) - 32, "sub-block {j}");
        }
    }

    #[test]
    fn q2_q3_reject_bad_input() {
        assert_eq!(
            dequantize_q2_k(&[0; 83]),
            Err(KQuantError::BadByteLength {
                len: 83,
                block_bytes: 84
            })
        );
        assert_eq!(
            dequantize_q3_k(&[0; 111]),
            Err(KQuantError::BadByteLength {
                len: 111,
                block_bytes: 110
            })
        );
        assert_eq!(
            quantize_q2_k(&[0.0; 5]),
            Err(KQuantError::NotBlockAligned { len: 5 })
        );
        assert_eq!(
            quantize_q3_k(&[0.0; 5]),
            Err(KQuantError::NotBlockAligned { len: 5 })
        );
    }
}
