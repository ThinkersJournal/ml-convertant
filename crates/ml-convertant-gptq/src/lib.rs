//! GPTQ int4 (AutoGPTQ v1 layout) unpack, pack, dequantize and conversion to the AWQ layout.
//!
//! # Layout
//!
//! A linear layer has `IC` input and `OC` output features, quantized in groups of `group_size` along `IC`.
//! **HuggingFace / AutoGPTQ** ([`HfGptq`]): `qweight [IC/8, OC]` as `u32`, where word `r` of column `o` holds
//! input features `8r..8r+8` of output feature `o`, nibble `k` being feature `8r + k` (lowest nibble first);
//! `qzeros [G, OC/8]` as `u32`, word `c` of group `g` holding output features `8c..8c+8`, nibble `k` being
//! feature `8c + k`, **stored minus one** (AutoGPTQ's `zeros -= 1` before packing); `scales [G, OC]` as f16;
//! and optionally `g_idx [IC]`, the group of each input feature (sequential, `r / group_size`, unless the
//! layer was quantized in activation order).
//!
//! **Logical** ([`LogicalGptq`]): one 4-bit code per `u8`, `q[IC, OC]`, and the *effective* zero points
//! `z[G, OC]`, which are the stored nibble plus one taken modulo 16, exactly as AutoGPTQ's forward pass
//! reads them (`zeros + 1`, then `& 15`). An element dequantizes to `(q - z) * s`, rounded to f16, which is
//! the same arithmetic as AWQ's, so a sequential-`g_idx` layer converts to [`LogicalAwq`] without any change
//! to a code, zero or scale.
//!
//! # Error
//!
//! Every conversion here moves codes, zeros and scales without changing them, so the error is exactly zero.
//! [`GptqToAwq`] proves it per layer rather than assuming it: it dequantizes the input and the output
//! independently and reports their difference (always `max_abs == 0`, bound `0`).

use half::f16;
use ml_convertant_awq::{AwqError, LogicalAwq, dequantize_logical as awq_dequantize};
use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

/// Why a GPTQ conversion failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GptqError {
    /// `group_size` was zero.
    BadGroupSize(usize),
    /// A dimension is not a multiple of what the packing requires.
    BadDim {
        /// Which dimension.
        what: &'static str,
        /// Its value.
        value: usize,
        /// What it must be a multiple of.
        multiple_of: usize,
    },
    /// A buffer has the wrong length.
    BadLen {
        /// Which buffer.
        what: &'static str,
        /// Its length.
        got: usize,
        /// The expected length.
        want: usize,
    },
    /// A code or zero point does not fit in 4 bits.
    CodeOutOfRange {
        /// Which buffer.
        what: &'static str,
        /// Index of the offending element.
        index: usize,
    },
    /// `g_idx` names a group that does not exist.
    BadGroupIndex {
        /// Input feature.
        row: usize,
        /// The group it names.
        group: i64,
    },
    /// The layer is in activation order (`g_idx` is not `r / group_size`); the conversion would also have to
    /// permute the activations, which is not a weight conversion.
    ActivationOrder,
    /// The AWQ layout refused the layer.
    Awq(AwqError),
    /// Measuring failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl fmt::Display for GptqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadGroupSize(g) => write!(f, "invalid group size {g}"),
            Self::BadDim {
                what,
                value,
                multiple_of,
            } => write!(f, "{what} = {value} is not a multiple of {multiple_of}"),
            Self::BadLen { what, got, want } => {
                write!(f, "{what} has {got} elements, expected {want}")
            }
            Self::CodeOutOfRange { what, index } => {
                write!(f, "{what}[{index}] does not fit in 4 bits")
            }
            Self::BadGroupIndex { row, group } => {
                write!(f, "g_idx[{row}] = {group} names no group")
            }
            Self::ActivationOrder => f.write_str(
                "activation-order layer (g_idx is not sequential): converting it needs the activations permuted",
            ),
            Self::Awq(e) => write!(f, "{e}"),
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for GptqError {}

/// A layer as stored in a HuggingFace `*-GPTQ` checkpoint (AutoGPTQ v1, 4-bit).
#[derive(Debug, Clone, PartialEq)]
pub struct HfGptq {
    /// Input features.
    pub ic: usize,
    /// Output features.
    pub oc: usize,
    /// Quantization group size along `IC`.
    pub group_size: usize,
    /// `[IC/8, OC]` packed weights.
    pub qweight: Vec<u32>,
    /// `[IC/group_size, OC/8]` packed zero points, stored minus one.
    pub qzeros: Vec<u32>,
    /// `[IC/group_size, OC]` scales.
    pub scales: Vec<f16>,
    /// `[IC]` group of each input feature, if the checkpoint carries one.
    pub g_idx: Option<Vec<i32>>,
}

/// A layer as logical 4-bit codes, one per `u8`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalGptq {
    /// Input features.
    pub ic: usize,
    /// Output features.
    pub oc: usize,
    /// Quantization group size along `IC`.
    pub group_size: usize,
    /// `[IC, OC]` codes, each in `0..16`.
    pub q: Vec<u8>,
    /// `[IC/group_size, OC]` effective zero points, each in `0..16`.
    pub z: Vec<u8>,
    /// `[IC/group_size, OC]` scales.
    pub scales: Vec<f16>,
    /// `[IC]` group of each input feature, if present.
    pub g_idx: Option<Vec<i32>>,
}

fn check_len(what: &'static str, got: usize, want: usize) -> Result<(), GptqError> {
    if got == want {
        Ok(())
    } else {
        Err(GptqError::BadLen { what, got, want })
    }
}

fn check_multiple(what: &'static str, value: usize, multiple_of: usize) -> Result<(), GptqError> {
    if multiple_of != 0 && value.is_multiple_of(multiple_of) {
        Ok(())
    } else {
        Err(GptqError::BadDim {
            what,
            value,
            multiple_of,
        })
    }
}

/// Validates dimensions; returns the group count.
fn geometry(ic: usize, oc: usize, group_size: usize) -> Result<usize, GptqError> {
    if group_size == 0 {
        return Err(GptqError::BadGroupSize(0));
    }
    check_multiple("IC", ic, group_size)?;
    check_multiple("IC", ic, 8)?;
    check_multiple("OC", oc, 8)?;
    Ok(ic / group_size)
}

fn check_g_idx(g_idx: &[i32], ic: usize, groups: usize) -> Result<(), GptqError> {
    check_len("g_idx", g_idx.len(), ic)?;
    match g_idx
        .iter()
        .position(|&g| usize::try_from(g).map_or(true, |g| g >= groups))
    {
        Some(row) => Err(GptqError::BadGroupIndex {
            row,
            group: i64::from(g_idx[row]),
        }),
        None => Ok(()),
    }
}

/// Whether `g_idx` is the sequential `r / group_size`.
fn is_sequential(g_idx: &[i32], group_size: usize) -> bool {
    g_idx
        .iter()
        .enumerate()
        .all(|(r, &g)| usize::try_from(g).is_ok_and(|g| g == r / group_size))
}

/// Unpacks a HuggingFace layer to logical codes.
pub fn unpack_hf(hf: &HfGptq) -> Result<LogicalGptq, GptqError> {
    let groups = geometry(hf.ic, hf.oc, hf.group_size)?;
    check_len("qweight", hf.qweight.len(), (hf.ic / 8) * hf.oc)?;
    check_len("qzeros", hf.qzeros.len(), groups * (hf.oc / 8))?;
    check_len("scales", hf.scales.len(), groups * hf.oc)?;
    if let Some(g) = &hf.g_idx {
        check_g_idx(g, hf.ic, groups)?;
    }
    let mut q = vec![0u8; hf.ic * hf.oc];
    for r in 0..hf.ic / 8 {
        for o in 0..hf.oc {
            let w = hf.qweight[r * hf.oc + o];
            for k in 0..8 {
                q[(r * 8 + k) * hf.oc + o] = ((w >> (4 * k)) & 0xF) as u8;
            }
        }
    }
    let mut z = vec![0u8; groups * hf.oc];
    for g in 0..groups {
        for c in 0..hf.oc / 8 {
            let w = hf.qzeros[g * (hf.oc / 8) + c];
            for k in 0..8 {
                // AutoGPTQ stores zeros - 1 and its forward pass reads (stored + 1) & 15.
                z[g * hf.oc + c * 8 + k] = ((((w >> (4 * k)) & 0xF) + 1) & 0xF) as u8;
            }
        }
    }
    Ok(LogicalGptq {
        ic: hf.ic,
        oc: hf.oc,
        group_size: hf.group_size,
        q,
        z,
        scales: hf.scales.clone(),
        g_idx: hf.g_idx.clone(),
    })
}

/// Packs logical codes the way AutoGPTQ's `QuantLinear.pack` does.
pub fn pack_hf(l: &LogicalGptq) -> Result<HfGptq, GptqError> {
    let groups = geometry(l.ic, l.oc, l.group_size)?;
    check_len("q", l.q.len(), l.ic * l.oc)?;
    check_len("z", l.z.len(), groups * l.oc)?;
    check_len("scales", l.scales.len(), groups * l.oc)?;
    if let Some(index) = l.q.iter().position(|&v| v > 15) {
        return Err(GptqError::CodeOutOfRange { what: "q", index });
    }
    if let Some(index) = l.z.iter().position(|&v| v > 15) {
        return Err(GptqError::CodeOutOfRange { what: "z", index });
    }
    if let Some(g) = &l.g_idx {
        check_g_idx(g, l.ic, groups)?;
    }
    let mut qweight = vec![0u32; (l.ic / 8) * l.oc];
    for r in 0..l.ic / 8 {
        for o in 0..l.oc {
            for k in 0..8 {
                qweight[r * l.oc + o] |= u32::from(l.q[(r * 8 + k) * l.oc + o]) << (4 * k);
            }
        }
    }
    let mut qzeros = vec![0u32; groups * (l.oc / 8)];
    for g in 0..groups {
        for c in 0..l.oc / 8 {
            for k in 0..8 {
                let stored = (u32::from(l.z[g * l.oc + c * 8 + k]) + 15) & 0xF;
                qzeros[g * (l.oc / 8) + c] |= stored << (4 * k);
            }
        }
    }
    Ok(HfGptq {
        ic: l.ic,
        oc: l.oc,
        group_size: l.group_size,
        qweight,
        qzeros,
        scales: l.scales.clone(),
        g_idx: l.g_idx.clone(),
    })
}

/// Dequantizes to `[IC, OC]` values: `(q - z) * s`, rounded to f16, with the group of each row taken from
/// `g_idx` when present.
pub fn dequantize_logical(l: &LogicalGptq) -> Result<Vec<f32>, GptqError> {
    let groups = geometry(l.ic, l.oc, l.group_size)?;
    check_len("q", l.q.len(), l.ic * l.oc)?;
    check_len("z", l.z.len(), groups * l.oc)?;
    check_len("scales", l.scales.len(), groups * l.oc)?;
    if let Some(g) = &l.g_idx {
        check_g_idx(g, l.ic, groups)?;
    }
    let mut out = Vec::with_capacity(l.ic * l.oc);
    for r in 0..l.ic {
        let g = l.g_idx.as_ref().map_or(r / l.group_size, |g| g[r] as usize);
        for o in 0..l.oc {
            let d = f32::from(l.q[r * l.oc + o]) - f32::from(l.z[g * l.oc + o]);
            out.push(f16::from_f32(d * l.scales[g * l.oc + o].to_f32()).to_f32());
        }
    }
    Ok(out)
}

/// The same layer in the AWQ logical layout, unchanged. Refuses an activation-order layer.
pub fn to_awq_logical(l: &LogicalGptq) -> Result<LogicalAwq, GptqError> {
    geometry(l.ic, l.oc, l.group_size)?;
    if l.g_idx
        .as_ref()
        .is_some_and(|g| !is_sequential(g, l.group_size))
    {
        return Err(GptqError::ActivationOrder);
    }
    Ok(LogicalAwq {
        ic: l.ic,
        oc: l.oc,
        group_size: l.group_size,
        q: l.q.clone(),
        z: l.z.clone(),
        scales: l.scales.clone(),
    })
}

/// HuggingFace GPTQ layer -> the same layer in the AWQ logical layout, reporting the (zero) error between
/// the dequantized weights before and after.
#[derive(Debug, Clone, Copy, Default)]
pub struct GptqToAwq;

impl Conversion for GptqToAwq {
    type Input = HfGptq;
    type Output = LogicalAwq;
    type Error = GptqError;

    fn convert(&self, input: &HfGptq) -> Result<Converted<LogicalAwq>, GptqError> {
        let logical = unpack_hf(input)?;
        let output = to_awq_logical(&logical)?;
        let before = dequantize_logical(&logical)?;
        let after = awq_dequantize(&output).map_err(GptqError::Awq)?;
        let report = ErrorReport::measure(&before, &after)
            .map_err(GptqError::Measure)?
            .with_bound(0.0);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests;
