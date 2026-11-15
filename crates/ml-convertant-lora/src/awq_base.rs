//! LoRA merge into an **AWQ** base: dequantize, merge in f32, requantize each group.
//!
//! The base is a [`LogicalAwq`] (4-bit codes, zero points and f16 scales; a HuggingFace layer is unpacked with
//! `ml_convertant_awq::unpack_hf` first and the result repacked with `pack_hf` or `repack_for_kernel`).
//! Codes are `[IC, OC]`; the adapter is as in [`crate::LoraAdapter`] with `in_features == ic` and
//! `out_features == oc`. A group is `group_size` consecutive input features of one output feature, which is a
//! contiguous run of a row of the `[out, in]` weight, so `in_features` must be a multiple of `group_size`.
//!
//! # What the requantizer is, and is not
//!
//! The merged weights are quantized group by group with plain asymmetric min/max round-to-nearest: the range
//! of the group is widened to include 0, the f16 scale is `range / 15` **rounded up** to the next f16, and the
//! zero point is the nearest integer to `-min / scale` (always in `0..=15`). It is **not** AWQ's
//! activation-aware scale search, which needs calibration data this crate does not have; a checkpoint merged
//! here is a valid AWQ-layout layer whose scales were chosen from the merged weights alone.
//!
//! # Error
//!
//! The reference is the exact f64 merge of the dequantized base; the report is the dequantized output
//! (`(q - z) * s` rounded to f16, as `dequantize_logical` computes it) against that reference. The bound is
//! [`crate::lora_merge_bound`] plus the largest per-group bound `s * (1/2 + 15 * 2^-11 + 2^-20) + 2^-25`:
//! with `s >= range / 15` and `z` the nearest integer, every value lies within `s / 2` of the code grid, and
//! the f16 rounding of `(q - z) * s` adds at most `2^-11` of a value of size at most `15 s`.

use crate::{
    LoraAdapter, LoraError, RequantError, lora_merge_bound, merge_lora, merge_lora_f64_reference,
};
use half::f16;
use ml_convertant_awq::{AwqError, LogicalAwq, dequantize_logical};
use ml_convertant_core::{Converted, ErrorReport};
use std::fmt;

/// Why quantizing the merged weights into the AWQ layout failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwqRequantError {
    /// The base layer is malformed (see [`AwqError`]).
    Awq(AwqError),
    /// A merged value is not finite, or a group's scale does not fit in f16.
    NonFinite {
        /// Index into the `[out, in]` weight of the offending value or group start.
        index: usize,
    },
}

impl fmt::Display for AwqRequantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Awq(e) => write!(f, "{e}"),
            Self::NonFinite { index } => write!(
                f,
                "merged weight at {index} is not finite or its scale overflows f16"
            ),
        }
    }
}

impl std::error::Error for AwqRequantError {}

fn awq(e: AwqError) -> RequantError<AwqRequantError> {
    RequantError::Format(AwqRequantError::Awq(e))
}

/// `[IC, OC]` dequantized values to the adapter's `[out, in]` layout.
fn to_out_in(deq: &[f32], ic: usize, oc: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; ic * oc];
    for i in 0..ic {
        for o in 0..oc {
            w[o * ic + i] = deq[i * oc + o];
        }
    }
    w
}

/// The smallest f16 that is `>= x` for finite positive `x`, or `None` if it overflows f16.
fn f16_round_up(x: f32) -> Option<f16> {
    let r = f16::from_f32(x);
    let r = if r.to_f32() < x {
        f16::from_bits(r.to_bits() + 1)
    } else {
        r
    };
    r.is_finite().then_some(r)
}

/// Quantizes `merged` (`[out, in]` row-major) into the AWQ layout with groups of `gs`, returning the layer and
/// the largest per-group error bound.
fn requantize(
    merged: &[f32],
    ic: usize,
    oc: usize,
    gs: usize,
) -> Result<(LogicalAwq, f64), AwqRequantError> {
    let groups = ic / gs;
    let mut q = vec![0u8; ic * oc];
    let mut z = vec![0u8; groups * oc];
    let mut scales = vec![f16::from_f32(1.0); groups * oc];
    let mut worst = 0.0f64;
    for o in 0..oc {
        for g in 0..groups {
            let run = &merged[o * ic + g * gs..o * ic + (g + 1) * gs];
            if let Some(i) = run.iter().position(|v| !v.is_finite()) {
                return Err(AwqRequantError::NonFinite {
                    index: o * ic + g * gs + i,
                });
            }
            let lo = run.iter().fold(0.0f32, |m, &v| m.min(v));
            let hi = run.iter().fold(0.0f32, |m, &v| m.max(v));
            let range = hi - lo;
            if range == 0.0 {
                continue; // all zero: code 0, zero 0, scale 1 decode to exactly 0
            }
            let s = f16_round_up(range / 15.0).ok_or(AwqRequantError::NonFinite {
                index: o * ic + g * gs,
            })?;
            let sf = s.to_f32();
            let zp = (-lo / sf).round().clamp(0.0, 15.0);
            z[g * oc + o] = zp as u8;
            scales[g * oc + o] = s;
            for (k, &v) in run.iter().enumerate() {
                q[(g * gs + k) * oc + o] = ((v / sf).round() + zp).clamp(0.0, 15.0) as u8;
            }
            let sd = f64::from(sf);
            worst = worst.max(sd * (0.5 + 15.0 * 2f64.powi(-11) + 2f64.powi(-20)) + 2f64.powi(-25));
        }
    }
    Ok((
        LogicalAwq {
            ic,
            oc,
            group_size: gs,
            q,
            z,
            scales,
        },
        worst,
    ))
}

/// Merges `adapter` into the AWQ base `base`, returning the requantized layer with the chain's error report.
pub fn merge_lora_into_awq(
    base: &LogicalAwq,
    adapter: &LoraAdapter,
) -> Result<Converted<LogicalAwq>, RequantError<AwqRequantError>> {
    let (ic, oc, gs) = (base.ic, base.oc, base.group_size);
    if gs == 0 || !adapter.in_features.is_multiple_of(gs) {
        return Err(RequantError::BlockSpansRows {
            in_features: adapter.in_features,
            block_elems: gs,
        });
    }
    if adapter.in_features != ic || adapter.out_features != oc {
        return Err(RequantError::Lora(LoraError::BadLen {
            what: "adapter dimensions against the base [IC, OC]",
            got: adapter.out_features * adapter.in_features,
            want: oc * ic,
        }));
    }
    let deq = dequantize_logical(base).map_err(awq)?;
    let w = to_out_in(&deq, ic, oc);
    let merged = merge_lora(&w, adapter).map_err(RequantError::Lora)?;
    let (output, requant) = requantize(&merged, ic, oc, gs).map_err(RequantError::Format)?;
    let decoded = to_out_in(&dequantize_logical(&output).map_err(awq)?, ic, oc);
    let reference = merge_lora_f64_reference(&w, adapter).map_err(RequantError::Lora)?;
    let bound = lora_merge_bound(&w, adapter).map_err(RequantError::Lora)? + requant;
    let report = ErrorReport::measure(&reference, &decoded)
        .map_err(RequantError::Measure)?
        .with_bound(bound);
    Ok(Converted { output, report })
}

#[cfg(test)]
mod tests;
