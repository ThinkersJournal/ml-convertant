//! IQ4_NL: 4-bit non-linear. 32 elements per block, 18 bytes: a little-endian IEEE half scale `d` followed
//! by 16 bytes of nibbles. Byte `j` holds element `j` in its low nibble and element `j + 16` in its high
//! nibble; a nibble `q` decodes to `d * KVALUES_IQ4NL[q]`, a fixed table of sixteen `i8` values.
//!
//! # Encoder
//!
//! The encoder is ggml's `quantize_iq4_nl` (the function `llama-quantize` uses, `ntry = 7`), reproduced
//! operation for operation in f32 against llama.cpp `9d57ce456c`: the per-element weight is `x^2`
//! (or `qw * sqrt(sigma2 + x^2)` with an importance vector `qw`), a first scale from the signed element of
//! largest magnitude, fifteen candidate scales `(itry + values[0]) / max` for `itry` in `-7..=7`, each scored
//! by `sumqx^2 / sumq2`, the winner's least-squares scale `sumqx / sumq2` stored as f16 (round to nearest
//! even), and the nibbles recomputed from that scale. Bytes are identical to ggml's; the golden test and the
//! CI differential check that.
//!
//! A block whose largest magnitude is under `1e-15` (an all-zero block, or a vanishingly small one) is stored
//! with scale 0 and decodes to zeros. Its nibbles are all 8 (`0 * x` maps to table entry 8, the value 1),
//! because ggml recomputes them from the inverse scale, which it takes as 0, for every block.
//!
//! # Error bound
//!
//! With `d32` the f32 scale the nibbles were chosen against, `d16` the stored scale, `t = x / d32`, and the
//! table spanning `[v0, v15] = [-127, 113]` with a largest gap of 24: the chosen value is the table entry
//! nearest `t`, so `|t - v| <= 12` inside the table's span and `|t - v| <= (v0 - t)` or `(t - v15)` outside
//! it. Then `|x - d16 * v| <= |d32| * |t - v| + 127 * |d32 - d16|`, plus the f32 rounding of `1 / d32`, of
//! the product `id * x` and of the final `d16 * v`, which [`iq4_nl_block_bound`] covers with a generous
//! `2^-22` slack. The weights are *not* uniform (they are `x^2` unless an importance vector is given), so the
//! scale may sit below `max / 127` and a large element can fall outside the table's span: the bound is
//! computed element by element from `t` and is larger there. An infinite `d16` gives an infinite bound.
//!
//! Non-finite inputs are errors, and so is a block whose scale or intermediate sums overflow f32 (inputs
//! above about 1e19 square past f32's range).

use half::f16;
use ml_convertant_core::{Conversion, Converted, ErrorReport};
use std::fmt;

use crate::BlockError;

/// Elements per IQ4_NL block.
pub const QK4_NL: usize = 32;
/// Bytes per IQ4_NL block: 2 (f16 scale) + 16 (nibbles).
pub const IQ4_NL_BLOCK_BYTES: usize = 2 + QK4_NL / 2;
/// ggml's `kvalues_iq4nl` (`ggml-common.h`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// ggml's `GROUP_MAX_EPS`: a block whose largest magnitude is below this is stored as zeros.
const GROUP_MAX_EPS: f32 = 1e-15;
/// ggml's `ntry` for `quantize_iq4_nl`.
const NTRY: i32 = 7;

/// Why an IQ4_NL conversion failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iq4Error {
    /// Length, finiteness or measurement problem shared with the other block formats.
    Block(BlockError),
    /// Block `block`'s scale or intermediate sums are not finite (inputs too large to square in f32).
    NonFiniteScale {
        /// Block index.
        block: usize,
    },
    /// The importance vector is empty, not a multiple of the block size, does not divide the input length
    /// into whole rows, or holds a non-finite or negative value.
    BadImportance {
        /// Importance vector length.
        len: usize,
        /// Input length.
        input_len: usize,
    },
}

impl From<BlockError> for Iq4Error {
    fn from(e: BlockError) -> Self {
        Self::Block(e)
    }
}

impl fmt::Display for Iq4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Block(e) => e.fmt(f),
            Self::NonFiniteScale { block } => {
                write!(
                    f,
                    "block {block} has a non-finite scale or overflowing sums"
                )
            }
            Self::BadImportance { len, input_len } => write!(
                f,
                "importance vector of {len} values does not fit rows of an input of {input_len} elements"
            ),
        }
    }
}

impl std::error::Error for Iq4Error {}

/// ggml's `best_index_int8` for the sixteen-entry table.
fn best_index(x: f32) -> usize {
    let val = &KVALUES_IQ4NL;
    if x <= f32::from(val[0]) {
        return 0;
    }
    if x >= f32::from(val[15]) {
        return 15;
    }
    let (mut ml, mut mu) = (0usize, 15usize);
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < f32::from(val[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - f32::from(val[mu - 1]) < f32::from(val[mu]) - x {
        mu - 1
    } else {
        mu
    }
}

/// One block: its f32 scale (the stored f16 is `f16::from_f32` of it), written into `l` (the nibbles, one per
/// element).
fn encode_block(xb: &[f32; QK4_NL], qw: Option<&[f32]>, l: &mut [u8; QK4_NL]) -> f32 {
    let mut weight = [0.0f32; QK4_NL];
    if let Some(qw) = qw {
        let mut sigma2 = 0.0f32;
        for &v in xb {
            sigma2 += v * v;
        }
        sigma2 *= 2.0f32 / 32.0f32;
        for j in 0..QK4_NL {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
    } else {
        for j in 0..QK4_NL {
            weight[j] = xb[j] * xb[j];
        }
    }
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in xb {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    let d = if amax < GROUP_MAX_EPS {
        0.0f32
    } else {
        scale_search(xb, &weight, max, l)
    };
    let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
    for j in 0..QK4_NL {
        l[j] = best_index(id * xb[j]) as u8;
    }
    d
}

/// ggml's scale search for a block that is not all zero: the least-squares scale of the best of the candidates.
fn scale_search(xb: &[f32; QK4_NL], weight: &[f32; QK4_NL], max: f32, l: &mut [u8; QK4_NL]) -> f32 {
    let v0 = f32::from(KVALUES_IQ4NL[0]);
    let d = -max / v0;
    let id = 1.0f32 / d;
    let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
    for j in 0..QK4_NL {
        let al = id * xb[j];
        let idx = best_index(al);
        l[j] = idx as u8;
        let q = f32::from(KVALUES_IQ4NL[idx]);
        let w = weight[j];
        sumqx += w * q * xb[j];
        sumq2 += w * q * q;
    }
    let mut d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0f32 };
    let mut best = d * sumqx;
    for itry in -NTRY..=NTRY {
        let id = (itry + i32::from(KVALUES_IQ4NL[0])) as f32 / max;
        let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
        for j in 0..QK4_NL {
            let al = id * xb[j];
            let idx = best_index(al);
            let q = f32::from(KVALUES_IQ4NL[idx]);
            let w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }
    d
}

/// Checks `importance` against the input and returns its row length, or `None` when there is none.
fn check_importance(x: &[f32], importance: Option<&[f32]>) -> Result<Option<usize>, Iq4Error> {
    let Some(w) = importance else { return Ok(None) };
    let bad = Iq4Error::BadImportance {
        len: w.len(),
        input_len: x.len(),
    };
    if w.is_empty()
        || !w.len().is_multiple_of(QK4_NL)
        || !x.len().is_multiple_of(w.len())
        || w.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(bad);
    }
    Ok(Some(w.len()))
}

/// Encodes `x` (a multiple of 32 finite values) to IQ4_NL bytes, with ggml's default (no importance vector).
pub fn quantize_iq4_nl(x: &[f32]) -> Result<Vec<u8>, Iq4Error> {
    quantize_iq4_nl_weighted(x, None)
}

/// Encodes `x` to IQ4_NL bytes with an optional importance vector, as ggml's `quantize_iq4_nl` takes
/// `quant_weights`: `importance` holds one non-negative weight per element of a row, `x` is whole rows of
/// that length, and the vector repeats for every row. Without it this is [`quantize_iq4_nl`].
pub fn quantize_iq4_nl_weighted(
    x: &[f32],
    importance: Option<&[f32]>,
) -> Result<Vec<u8>, Iq4Error> {
    if !x.len().is_multiple_of(QK4_NL) {
        return Err(BlockError::NotBlockAligned { len: x.len() }.into());
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(BlockError::NonFinite { index }.into());
    }
    let row_len = check_importance(x, importance)?;
    let mut out = Vec::with_capacity(x.len() / QK4_NL * IQ4_NL_BLOCK_BYTES);
    let mut l = [0u8; QK4_NL];
    for (block, xb) in x.as_chunks::<QK4_NL>().0.iter().enumerate() {
        let qw = importance.zip(row_len).map(|(w, row)| {
            let start = (block * QK4_NL) % row;
            &w[start..start + QK4_NL]
        });
        let d = encode_block(xb, qw, &mut l);
        if !d.is_finite() {
            return Err(Iq4Error::NonFiniteScale { block });
        }
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..QK4_NL / 2 {
            out.push(l[j] | (l[j + QK4_NL / 2] << 4));
        }
    }
    Ok(out)
}

/// Decodes IQ4_NL bytes to f32.
pub fn dequantize_iq4_nl(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ4_NL_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ4_NL_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ4_NL_BLOCK_BYTES * QK4_NL];
    for (block, dst) in bytes
        .as_chunks::<IQ4_NL_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK4_NL>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for (j, &q) in block[2..].iter().enumerate() {
            dst[j] = d * f32::from(KVALUES_IQ4NL[usize::from(q & 0x0F)]);
            dst[j + QK4_NL / 2] = d * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
        }
    }
    Ok(out)
}

/// The proven worst-case `|x - x'|` for one block given the scale `d32` it was encoded against (see the
/// module docs). `d32` is the f32 scale the encoder returned for this block; use [`Iq4Nl`] to get the bound
/// of a whole conversion.
pub fn iq4_nl_block_bound(chunk: &[f32; QK4_NL], d32: f32) -> f64 {
    let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if d32 == 0.0 {
        return f64::from(amax);
    }
    let d16 = f16::from_f32(d32).to_f32();
    if d16.is_infinite() {
        return f64::INFINITY;
    }
    let (d32, d16) = (f64::from(d32), f64::from(d16));
    let (v0, v15) = (f64::from(KVALUES_IQ4NL[0]), f64::from(KVALUES_IQ4NL[15]));
    let slack = 2f64.powi(-22);
    let mut worst = 0.0f64;
    for &x in chunk {
        let t = f64::from(x) / d32;
        let e = if t < v0 {
            v0 - t
        } else if t > v15 {
            t - v15
        } else {
            12.0
        };
        let b = d32.abs() * (e + (t.abs() + 127.0) * slack)
            + 127.0 * (d32 - d16).abs()
            + f64::from(x).abs() * 2f64.powi(-23);
        worst = worst.max(b);
    }
    worst
}

/// f32 -> IQ4_NL, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Iq4Nl<'a> {
    /// Optional importance vector (see [`quantize_iq4_nl_weighted`]).
    pub importance: Option<&'a [f32]>,
}

impl Conversion for Iq4Nl<'_> {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = Iq4Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, Iq4Error> {
        let output = quantize_iq4_nl_weighted(input, self.importance)?;
        let decoded = dequantize_iq4_nl(&output)?;
        // The bound needs each block's f32 scale, which the bytes only hold rounded to f16, so encode again
        // keeping it (the encoding is deterministic).
        let row_len = check_importance(input, self.importance)?;
        let mut l = [0u8; QK4_NL];
        let mut bound = 0.0f64;
        for (block, xb) in input.as_chunks::<QK4_NL>().0.iter().enumerate() {
            let qw = self.importance.zip(row_len).map(|(w, row)| {
                let start = (block * QK4_NL) % row;
                &w[start..start + QK4_NL]
            });
            let d = encode_block(xb, qw, &mut l);
            bound = bound.max(iq4_nl_block_bound(xb, d));
        }
        let report = ErrorReport::measure(input, &decoded)
            .map_err(BlockError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests;
