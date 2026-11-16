//! IQ3_S: 3.4375 bits per weight from a fixed grid of 512 four-element points. 256 elements per block, 110
//! bytes: an f16 `d`, `qs[64]` (low eight bits of a grid index per four elements), `qh[8]` (the ninth bit; bit `k`
//! of `qh[ib]` belongs to group `k` of sub-block `ib`), `signs[32]` (one bit per element, set = negative) and
//! `scales[4]` (a 4-bit scale `l` per 32-element sub-block, low nibble for an even sub-block). An element decodes
//! to `d * (1 + 2 * l) * g * s`, with `g` the grid byte (an odd number 1..=15) and `s` the sign.
//!
//! The grid is ggml's `iq3s_grid` (`iq3s_grid.rs`, generated from the pinned llama.cpp by
//! `tools/golden/gen_iq3s_grid.py`, which checks it against ggml's `kgrid_512` as well; CI regenerates and diffs).
//!
//! # Encoder
//!
//! ggml's `quantize_iq3_s` (`quantize_row_iq3_s_impl`, 32-element sub-blocks, `kMaxQ = 8`) reproduced in f32
//! against llama.cpp `9d57ce456c`: per sub-block, 19 candidate scales `(15 + 0.2 * is) / max` for `is` in
//! `-9..=9`, each element rounded to a level `0..=7`, a four-element group that is not a grid point replaced by
//! the nearest grid point among those within the three smallest distinct distances (ggml's neighbour lists, rebuilt
//! here from the grid by the same procedure), the best candidate by `sumqx^2 / sumq2`, a refinement pass,
//! then the block scale `d = max_scale / 31` stored as f16 of `d * 1.033`, and the sub-block scales
//! `l = clamp(round_ties_even((scale / d - 1) / 2), 0, 15)`. Weights are `x^2`, or `qw * sqrt(sigma2 + x^2)` with
//! `sigma2` over the 256-element block.
//!
//! ## One deliberate difference from ggml
//!
//! When a 32-element sub-block is exactly zero but a later one is not, ggml (`9d57ce456c`) does `continue` before
//! it advances its output pointers to `qs` and `signs`, so every later sub-block's grid indices and signs are
//! written one slot too early and the block decodes to the wrong values (the scales and `qh` stay in place).
//! This encoder writes each sub-block where the decoder reads it. The bytes are therefore identical to ggml's
//! on every input without an exactly-zero sub-block next to a non-zero one, which the golden vectors and the CI
//! differential cover, and differ on those blocks, where ggml's own decode does not reproduce its input.
//! An exactly-zero sub-block decodes to `d` (IQ3_S has no zero: the smallest magnitude is `d * 1 * 1`).
//!
//! # Error
//!
//! Measured only: the nearest-neighbour choice among the grid's points is not a nearest-point search, so no
//! bound is claimed.

use half::f16;
use ml_convertant_core::{Conversion, Converted, ErrorReport};
use std::sync::OnceLock;

use crate::BlockError;
use crate::iq3s_grid::IQ3S_GRID;
use crate::iq4_nl::Iq4Error;
use crate::iq4_xs::check_importance;

/// Elements per IQ3_S block (ggml's `QK_K`).
pub const QK_IQ3_S: usize = 256;
/// Bytes per IQ3_S block: 2 (`d`) + 64 (`qs`) + 8 (`qh`) + 32 (`signs`) + 4 (`scales`).
pub const IQ3_S_BLOCK_BYTES: usize = 2 + 64 + 8 + 32 + 4;

const SUB: usize = 32;
const NSUB: usize = QK_IQ3_S / SUB;
const KMAXQ: i32 = 8;
/// ggml's number of distinct distances kept in a neighbour list for the 512-point grid.
const NWANT: usize = 3;

struct Tables {
    /// For each of the 4096 packed level-quadruples, the grid index, or `-1`.
    map: Vec<i32>,
    /// For a quadruple that is not a grid point: the grid indices within the `NWANT` smallest distinct distances,
    /// ordered by (squared distance, index) as ggml's `qsort` orders them.
    neighbours: Vec<Vec<u16>>,
}

fn point(j: usize) -> [i32; 4] {
    let b = IQ3S_GRID[j].to_le_bytes();
    [
        i32::from(b[0]),
        i32::from(b[1]),
        i32::from(b[2]),
        i32::from(b[3]),
    ]
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut map = vec![-1i32; 4096];
        for j in 0..IQ3S_GRID.len() {
            let p = point(j);
            let idx: usize = (0..4).map(|k| (((p[k] - 1) / 2) as usize) << (3 * k)).sum();
            map[idx] = j as i32;
        }
        let mut neighbours = vec![Vec::new(); 4096];
        for (i, nb) in neighbours.iter_mut().enumerate() {
            if map[i] >= 0 {
                continue;
            }
            let pos: [i32; 4] = std::array::from_fn(|k| 2 * ((i >> (3 * k)) & 7) as i32 + 1);
            let mut d2: Vec<(i32, u16)> = (0..IQ3S_GRID.len())
                .map(|j| {
                    let p = point(j);
                    (
                        (0..4).map(|k| (p[k] - pos[k]) * (p[k] - pos[k])).sum(),
                        j as u16,
                    )
                })
                .collect();
            d2.sort_unstable();
            let mut cur = d2[0].0;
            let mut have = 1;
            for &(d, j) in &d2 {
                if d > cur {
                    if have == NWANT {
                        break;
                    }
                    cur = d;
                    have += 1;
                }
                nb.push(j);
            }
        }
        Tables { map, neighbours }
    })
}

/// ggml's `iq3_find_best_neighbour`: the neighbour with the smallest weighted distance (first on ties); sets the
/// four levels `l` and returns the grid index.
fn best_neighbour(nb: &[u16], xval: &[f32], weight: &[f32], scale: f32, l: &mut [i32]) -> usize {
    let mut best_d2 = f32::MAX;
    let mut best = usize::MAX;
    for &j in nb {
        let p = point(usize::from(j));
        let mut d2 = 0.0f32;
        for i in 0..4 {
            let q = p[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = usize::from(j);
        }
    }
    let p = point(best);
    for i in 0..4 {
        l[i] = (p[i] - 1) / 2;
    }
    best
}

fn level(id: f32, x: f32) -> i32 {
    ((0.5f32 * (id * x - 1.0)).round_ties_even() as i32).clamp(0, KMAXQ - 1)
}

fn pack(l: &[i32]) -> usize {
    (0..4).map(|i| (l[i] as usize) << (3 * i)).sum()
}

/// Encodes one 256-element block into 110 bytes appended to `out`.
fn encode_block(xbl: &[f32; QK_IQ3_S], qw: Option<&[f32]>, out: &mut Vec<u8>) -> Result<(), ()> {
    let t = tables();
    let mut qs = [0u8; 64];
    let mut qh = [0u8; 8];
    let mut signs = [0u8; 32];
    let mut scales = [0.0f32; NSUB];
    let mut sumx2 = 0.0f32;
    for &v in xbl {
        sumx2 += v * v;
    }
    let sigma2 = 2.0f32 * sumx2 / 256.0f32;
    let mut max_scale = 0.0f32;
    for ib in 0..NSUB {
        let xb = &xbl[ib * SUB..(ib + 1) * SUB];
        let mut weight = [0.0f32; SUB];
        match qw {
            Some(qw) => {
                for i in 0..SUB {
                    weight[i] = qw[ib * SUB + i] * (sigma2 + xb[i] * xb[i]).sqrt();
                }
            }
            None => {
                for i in 0..SUB {
                    weight[i] = xb[i] * xb[i];
                }
            }
        }
        let mut waux = [0.0f32; SUB];
        for i in 0..SUB {
            waux[i] = weight[i].sqrt();
        }
        let mut xval = [0.0f32; SUB];
        let mut block_signs = [0u8; SUB / 8];
        for k in 0..SUB / 8 {
            let mut s = 0u8;
            for i in 0..8 {
                let v = xb[8 * k + i];
                if v >= 0.0 {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= 1 << i;
                }
            }
            block_signs[k] = s;
        }
        let mut max = xval[0];
        for &v in &xval[1..] {
            max = max.max(v);
        }
        let mut l = [0i32; SUB];
        if max == 0.0 {
            scales[ib] = 0.0;
            // An all-zero sub-block: grid point 0, zero signs; the bytes stay zero.
            continue;
        }
        let mut best = 0.0f32;
        let mut scale = max / (2 * KMAXQ - 1) as f32;
        let mut on_grid = [false; SUB / 4];
        let mut laux = [0i32; SUB];
        let mut on_grid_aux = [false; SUB / 4];
        for is in -9..=9i32 {
            let id = ((2 * KMAXQ - 1) as f32 + is as f32 * 0.2f32) / max;
            let this_scale = 1.0f32 / id;
            for k in 0..SUB / 4 {
                for i in 0..4 {
                    laux[4 * k + i] = level(id, xval[4 * k + i]);
                }
                let u = pack(&laux[4 * k..4 * k + 4]);
                on_grid_aux[k] = true;
                if t.map[u] < 0 {
                    on_grid_aux[k] = false;
                    best_neighbour(
                        &t.neighbours[u],
                        &xval[4 * k..4 * k + 4],
                        &waux[4 * k..4 * k + 4],
                        this_scale,
                        &mut laux[4 * k..4 * k + 4],
                    );
                }
            }
            let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
            for i in 0..SUB {
                let w = weight[i];
                let q = (2 * laux[i] + 1) as f32;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l = laux;
                on_grid = on_grid_aux;
            }
        }
        if on_grid.iter().any(|g| !g) && scale > 0.0 {
            let id = 1.0f32 / scale;
            for k in 0..SUB / 4 {
                let mut lq = [0i32; 4];
                for i in 0..4 {
                    lq[i] = level(id, xval[4 * k + i]);
                }
                let u = pack(&lq);
                let gi = if t.map[u] >= 0 {
                    t.map[u] as usize
                } else {
                    best_neighbour(
                        &t.neighbours[u],
                        &xval[4 * k..4 * k + 4],
                        &waux[4 * k..4 * k + 4],
                        scale,
                        &mut lq,
                    )
                };
                let p = point(gi);
                for i in 0..4 {
                    l[4 * k + i] = (p[i] - 1) / 2;
                }
            }
            let (mut sumqx, mut sumq2) = (0.0f32, 0.0f32);
            for i in 0..SUB {
                let w = weight[i];
                let q = (2 * l[i] + 1) as f32;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut block_signs {
                *s = !*s;
            }
        }
        for k in 0..SUB / 4 {
            let u = pack(&l[4 * k..4 * k + 4]);
            let gi = t.map[u];
            if gi < 0 {
                return Err(());
            }
            let gi = gi as usize;
            qs[ib * 8 + k] = (gi & 255) as u8;
            qh[ib] |= ((gi >> 8) as u8) << k;
        }
        signs[ib * 4..ib * 4 + 4].copy_from_slice(&block_signs);
        if !scale.is_finite() || scale < 0.0 {
            return Err(());
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    let mut sc = [0u8; NSUB / 2];
    let mut d16 = f16::from_f32(0.0);
    if max_scale != 0.0 {
        let d = max_scale / 31.0f32;
        d16 = f16::from_f32(d * 1.033f32);
        let id = 1.0f32 / d;
        if !id.is_finite() {
            return Err(());
        }
        for ib in (0..NSUB).step_by(2) {
            let l1 = ((0.5f32 * (id * scales[ib] - 1.0)).round_ties_even() as i32).clamp(0, 15);
            let l2 = ((0.5f32 * (id * scales[ib + 1] - 1.0)).round_ties_even() as i32).clamp(0, 15);
            sc[ib / 2] = (l1 | (l2 << 4)) as u8;
        }
    }
    out.extend_from_slice(&d16.to_bits().to_le_bytes());
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&signs);
    out.extend_from_slice(&sc);
    Ok(())
}

/// Encodes `x` (a multiple of 256 finite values) to IQ3_S bytes, with ggml's default (no importance vector).
pub fn quantize_iq3_s(x: &[f32]) -> Result<Vec<u8>, Iq4Error> {
    quantize_iq3_s_weighted(x, None)
}

/// Encodes `x` to IQ3_S bytes with an optional importance vector, as ggml's `quantize_iq3_s` takes
/// `quant_weights`: one non-negative weight per element of a row (a multiple of 256 long); `x` is whole rows of
/// that length and the vector repeats for every row.
pub fn quantize_iq3_s_weighted(x: &[f32], importance: Option<&[f32]>) -> Result<Vec<u8>, Iq4Error> {
    if !x.len().is_multiple_of(QK_IQ3_S) {
        return Err(BlockError::NotBlockAligned { len: x.len() }.into());
    }
    if let Some(index) = x.iter().position(|v| !v.is_finite()) {
        return Err(BlockError::NonFinite { index }.into());
    }
    let row_len = check_importance(x, importance)?;
    let mut out = Vec::with_capacity(x.len() / QK_IQ3_S * IQ3_S_BLOCK_BYTES);
    for (block, xb) in x.as_chunks::<QK_IQ3_S>().0.iter().enumerate() {
        let qw = importance.zip(row_len).map(|(w, row)| {
            let start = (block * QK_IQ3_S) % row;
            &w[start..start + QK_IQ3_S]
        });
        encode_block(xb, qw, &mut out).map_err(|()| Iq4Error::NonFiniteScale { block })?;
    }
    Ok(out)
}

/// Decodes IQ3_S bytes to f32.
pub fn dequantize_iq3_s(bytes: &[u8]) -> Result<Vec<f32>, Iq4Error> {
    if !bytes.len().is_multiple_of(IQ3_S_BLOCK_BYTES) {
        return Err(BlockError::BadByteLength {
            len: bytes.len(),
            block_bytes: IQ3_S_BLOCK_BYTES,
        }
        .into());
    }
    let mut out = vec![0.0f32; bytes.len() / IQ3_S_BLOCK_BYTES * QK_IQ3_S];
    for (block, dst) in bytes
        .as_chunks::<IQ3_S_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK_IQ3_S>().0)
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..66];
        let qh = &block[66..74];
        let signs = &block[74..106];
        let scales = &block[106..110];
        for ib in 0..NSUB {
            let nib = if ib % 2 == 0 {
                scales[ib / 2] & 0xf
            } else {
                scales[ib / 2] >> 4
            };
            let db = d * (1 + 2 * i32::from(nib)) as f32;
            for l in 0..4 {
                let g1 = IQ3S_GRID[usize::from(qs[ib * 8 + 2 * l])
                    | ((usize::from(qh[ib]) << (8 - 2 * l)) & 256)]
                    .to_le_bytes();
                let g2 = IQ3S_GRID[usize::from(qs[ib * 8 + 2 * l + 1])
                    | ((usize::from(qh[ib]) << (7 - 2 * l)) & 256)]
                    .to_le_bytes();
                let s = signs[ib * 4 + l];
                for j in 0..4 {
                    let neg1 = s & (1 << j) != 0;
                    let neg2 = s & (1 << (j + 4)) != 0;
                    dst[ib * SUB + 8 * l + j] =
                        db * f32::from(g1[j]) * if neg1 { -1.0 } else { 1.0 };
                    dst[ib * SUB + 8 * l + j + 4] =
                        db * f32::from(g2[j]) * if neg2 { -1.0 } else { 1.0 };
                }
            }
        }
    }
    Ok(out)
}

/// f32 -> IQ3_S, reporting the measured error of the round trip against the input (no proven bound).
#[derive(Debug, Clone, Copy, Default)]
pub struct Iq3S<'a> {
    /// Optional importance vector (see [`quantize_iq3_s_weighted`]).
    pub importance: Option<&'a [f32]>,
}

impl Conversion for Iq3S<'_> {
    type Input = [f32];
    type Output = Vec<u8>;
    type Error = Iq4Error;

    fn convert(&self, input: &[f32]) -> Result<Converted<Vec<u8>>, Iq4Error> {
        let output = quantize_iq3_s_weighted(input, self.importance)?;
        let decoded = dequantize_iq3_s(&output)?;
        let report = ErrorReport::measure(input, &decoded).map_err(BlockError::Measure)?;
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests;
