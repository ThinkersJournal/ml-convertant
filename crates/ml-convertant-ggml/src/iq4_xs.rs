//! IQ4_XS: IQ4_NL's table with a two-level scale. 256 elements per block, 136 bytes: a little-endian f16 `d`, a
//! `u16` `scales_h`, `scales_l[4]` and `qs[128]`. Sub-block `ib` (32 elements) has a 6-bit scale `ls` whose low
//! four bits are the low (`ib` even) or high (`ib` odd) nibble of `scales_l[ib / 2]` and whose top two bits
//! are bits `2 * ib` and `2 * ib + 1` of `scales_h`; it decodes `d * (ls - 32) * KVALUES_IQ4NL[q]`. The nibbles
//! are laid out as in IQ4_NL (byte `j` of a sub-block: element `j` low, element `j + 16` high).
//!
//! # Encoder
//!
//! ggml's `quantize_iq4_xs` (which calls `quantize_row_iq4_nl_impl` with `QK_K`, 32, `ntry = 7`), reproduced in
//! f32 against llama.cpp `9d57ce456c`. Each sub-block gets IQ4_NL's scale search (weights `x^2`, or
//! `qw * sqrt(sigma2 + x^2)` with `sigma2` taken over the whole 256-element block); then the sub-block scale of
//! largest magnitude sets `d = -max_scale / 32`, each scale becomes `l = round_ties_even(scale / d)` clamped to
//! `-32..=31`, and the nibbles are chosen again against the *unrounded* product `d * l`. `d` is stored as f16.
//! A sub-block whose largest magnitude is under `1e-15` gets `l = 0` and all nibbles 8 (decoding to zero).
//!
//! # Error bound
//!
//! Per sub-block as for IQ4_NL ([`crate::iq4_nl_block_bound`]) with encoding scale `d32 * l` and decoding scale
//! `d16 * l`. Not-finite intermediate scales are errors.

use half::f16;
use ml_convertant_core::{Conversion, Converted, ErrorReport};

use crate::BlockError;
use crate::iq4_nl::{
    GROUP_MAX_EPS, Iq4Error, KVALUES_IQ4NL, QK4_NL, best_index, scale_search, sub_block_bound,
};

/// Elements per IQ4_XS block (ggml's `QK_K`).
pub const QK_IQ4_XS: usize = 256;
/// Sub-blocks per IQ4_XS block.
const NSUB: usize = QK_IQ4_XS / QK4_NL;
/// Bytes per IQ4_XS block: 2 (`d`) + 2 (`scales_h`) + 4 (`scales_l`) + 128 (nibbles).
pub const IQ4_XS_BLOCK_BYTES: usize = 2 + 2 + QK_IQ4_XS / 64 + QK_IQ4_XS / 2;

/// One encoded super-block, before packing.
struct Encoded {
    /// The f32 `d` (stored as f16).
    d: f32,
    /// Signed sub-block scales `-32..=31`.
    l: [i32; NSUB],
    /// Nibbles, one per element.
    q: [u8; QK_IQ4_XS],
}

fn encode_block(xb: &[f32; QK_IQ4_XS], qw: Option<&[f32]>) -> Result<Encoded, ()> {
    let mut sigma2 = 0.0f32;
    for &v in xb {
        sigma2 += v * v;
    }
    sigma2 *= 2.0f32 / 256.0f32;
    let mut scales = [0.0f32; NSUB];
    let (mut max_scale, mut amax_scale) = (0.0f32, 0.0f32);
    let mut scratch = [0u8; QK4_NL];
    for ib in 0..NSUB {
        let xs: &[f32; QK4_NL] = xb[ib * QK4_NL..(ib + 1) * QK4_NL]
            .try_into()
            .map_err(|_| ())?;
        let mut weight = [0.0f32; QK4_NL];
        if let Some(qw) = qw {
            for j in 0..QK4_NL {
                weight[j] = qw[ib * QK4_NL + j] * (sigma2 + xs[j] * xs[j]).sqrt();
            }
        } else {
            for j in 0..QK4_NL {
                weight[j] = xs[j] * xs[j];
            }
        }
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in xs {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let d = scale_search(xs, &weight, max, &mut scratch);
        scales[ib] = d;
        if !d.is_finite() {
            return Err(());
        }
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }
    let d = -max_scale / 32.0;
    let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
    if !d.is_finite() || !id.is_finite() {
        return Err(());
    }
    let mut out = Encoded {
        d,
        l: [0; NSUB],
        q: [0; QK_IQ4_XS],
    };
    for ib in 0..NSUB {
        let l = (id * scales[ib]).round_ties_even();
        if !l.is_finite() {
            return Err(());
        }
        let l = (l as i32).clamp(-32, 31);
        let dl = d * l as f32;
        let idl = if dl != 0.0 { 1.0f32 / dl } else { 0.0f32 };
        for j in 0..QK4_NL {
            out.q[ib * QK4_NL + j] = best_index(idl * xb[ib * QK4_NL + j]) as u8;
        }
        out.l[ib] = l;
    }
    Ok(out)
}

/// Checks `importance`: non-empty, a multiple of 256, dividing the input into whole rows, finite and non-negative.
fn check_importance(x: &[f32], importance: Option<&[f32]>) -> Result<Option<usize>, Iq4Error> {
    let Some(w) = importance else { return Ok(None) };
    if w.is_empty()
        || !w.len().is_multiple_of(QK_IQ4_XS)
        || !x.len().is_multiple_of(w.len())
        || w.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(Iq4Error::BadImportance {
            len: w.len(),
            input_len: x.len(),
        });
    }
    Ok(Some(w.len()))
}

fn pack(e: &Encoded, out: &mut Vec<u8>) {
    out.extend_from_slice(&f16::from_f32(e.d).to_bits().to_le_bytes());
    let mut scales_h = 0u16;
    let mut scales_l = [0u8; NSUB / 2];
    for ib in 0..NSUB {
        let l = (e.l[ib] + 32) as u8;
        let (lo, hi) = (l & 0xf, l >> 4);
        if ib % 2 == 0 {
            scales_l[ib / 2] = lo;
        } else {
            scales_l[ib / 2] |= lo << 4;
        }
        scales_h |= u16::from(hi) << (2 * ib);
    }
    out.extend_from_slice(&scales_h.to_le_bytes());
    out.extend_from_slice(&scales_l);
    for ib in 0..NSUB {
        for j in 0..QK4_NL / 2 {
            out.push(e.q[ib * QK4_NL + j] | (e.q[ib * QK4_NL + j + QK4_NL / 2] << 4));
        }
    }
}

/// Encodes `x` (a multiple of 256 finite values) to IQ4_XS bytes, with ggml's default (no importance vector).
pub fn quantize_iq4_xs(x: &[f32]) -> Result<Vec<u8>, Iq4Error> {
    quantize_iq4_xs_weighted(x, None)
}

/// Encodes `x` to IQ4_XS bytes with an optional importance vector, as ggml's `quantize_iq4_xs` takes
/// `quant_weights`: one non-negative weight per element of a row (a multiple of 256 long); `x` is whole rows of
/// that length and the vector repeats for every row.
pub fn quantize_iq4_xs_weighted(
    x: &[f32],
    importance: Option<&[f32]>,
) -> Result<Vec<u8>, Iq4Error> {
    Ok(encode_all(x, importance)?.0)
}

/// Bytes and the largest per-block bound.
fn encode_all(x: &[f32], importance: Option<&[f32]>) -> Result<(Vec<u8>, f64), Iq4Error> {
    if !x.len().is_multiple_of(QK_IQ4_XS) {
        return Err(BlockError::NotBlockAligned { len: x.len() }.into());
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(BlockError::NonFinite { index }.into());
    }
    let row_len = check_importance(x, importance)?;
    let mut out = Vec::with_capacity(x.len() / QK_IQ4_XS * IQ4_XS_BLOCK_BYTES);
    let mut bound = 0.0f64;
    for (block, xb) in x.as_chunks::<QK_IQ4_XS>().0.iter().enumerate() {
        let qw = importance.zip(row_len).map(|(w, row)| {
            let start = (block * QK_IQ4_XS) % row;
            &w[start..start + QK_IQ4_XS]
        });
        let e = encode_block(xb, qw).map_err(|()| Iq4Error::NonFiniteScale { block })?;
        pack(&e, &mut out);
        bound = bound.max(iq4_xs_block_bound(xb, &e));
    }
    Ok((out, bound))
}

fn iq4_xs_block_bound(xb: &[f32; QK_IQ4_XS], e: &Encoded) -> f64 {
    let d16 = f16::from_f32(e.d).to_f32();
    let mut worst = 0.0f64;
    for ib in 0..NSUB {
        let sub: &[f32; QK4_NL] = xb[ib * QK4_NL..(ib + 1) * QK4_NL]
            .try_into()
            .expect("32 elements");
        let l = e.l[ib] as f32;
        worst = worst.max(sub_block_bound(sub, e.d * l, d16 * l));
    }
    worst
}

/// Decodes IQ4_XS bytes to f32.
pub fn dequantize_iq4_xs(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ4_XS_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ4_XS_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ4_XS_BLOCK_BYTES * QK_IQ4_XS];
    for (block, dst) in bytes
        .as_chunks::<IQ4_XS_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ4_XS>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..];
        for ib in 0..NSUB {
            let ls = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                | (i32::from((scales_h >> (2 * ib)) & 3) << 4);
            let dl = d * (ls - 32) as f32;
            for j in 0..QK4_NL / 2 {
                let q = qs[ib * 16 + j];
                dst[ib * QK4_NL + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(q & 0x0F)]);
                dst[ib * QK4_NL + j + QK4_NL / 2] =
                    dl * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
            }
        }
    }
    Ok(out)
}

/// f32 -> IQ4_XS, reporting the error of the round trip against the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Iq4Xs<'a> {
    /// Optional importance vector (see [`quantize_iq4_xs_weighted`]).
    pub importance: Option<&'a [f32]>,
}

impl Conversion for Iq4Xs<'_> {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = Iq4Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, Iq4Error> {
        let (output, bound) = encode_all(input, self.importance)?;
        let decoded = dequantize_iq4_xs(&output)?;
        let report = ErrorReport::measure(input, &decoded)
            .map_err(BlockError::Measure)?
            .with_bound(bound);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests;
