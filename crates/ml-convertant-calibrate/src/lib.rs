//! Calibrated quantization into the AWQ 4-bit layout, starting with the round-to-nearest baseline.
//!
//! Weights are `[OC, IC]` row-major (the layout `ml-convertant-lora` uses); the result is a
//! [`LogicalAwq`] (`q[IC, OC]`, `z` and `scales` `[IC/group_size, OC]`). A group is `group_size`
//! consecutive input features of one output feature.
//!
//! # Round to nearest ([`rtn_awq`])
//!
//! Per group: the range is widened to include 0, the f16 scale is `range / 15` rounded **up** to the next f16,
//! the zero point is the nearest integer to `-min / scale` (always in `0..=15`), and each code is
//! `round(v / scale) + zero` clamped to `0..=15`. An all-zero group is code 0, zero 0, scale 1.
//!
//! # Error
//!
//! `report` compares the dequantized weights (`(q - z) * s` rounded to f16, as `dequantize_logical` computes
//! them) with the input; its bound is the largest per-group bound `s * (1/2 + 15 * 2^-11 + 2^-20) + 2^-25`
//! (`s >= range / 15` and `z` the nearest integer put every value within `s / 2` of the code grid; the f16
//! rounding adds at most `2^-11` of a value of size at most `15 s`). `output_report`, present only when a
//! [`CalibrationSet`] is given, compares `X W^T` with `X W'^T` (f64 accumulation); it is **measured, not
//! bounded**.

use half::f16;
use ml_convertant_awq::{AwqError, LogicalAwq, dequantize_logical};
use ml_convertant_core::{ErrorReport, LengthMismatch};
use std::fmt;

/// Why calibrated quantization failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrateError {
    /// `group_size` was zero.
    BadGroupSize(usize),
    /// A dimension is not usable (zero, or `ic` not a multiple of `group_size`).
    BadDim {
        /// Which dimension.
        what: &'static str,
        /// Its value.
        value: usize,
    },
    /// A buffer has the wrong length.
    BadLen {
        /// Which buffer.
        what: &'static str,
        /// Length given.
        got: usize,
        /// Length required.
        want: usize,
    },
    /// A weight or activation is not finite, or a group's scale does not fit in f16.
    NonFinite {
        /// Index into the offending buffer.
        index: usize,
    },
    /// The quantized layer was malformed (cannot happen for our own output).
    Awq(AwqError),
    /// Error measurement failed.
    Measure(LengthMismatch),
}

impl fmt::Display for CalibrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadGroupSize(g) => write!(f, "group size {g} is not usable"),
            Self::BadDim { what, value } => write!(f, "{what} = {value} is not usable"),
            Self::BadLen { what, got, want } => {
                write!(f, "{what} has {got} values, expected {want}")
            }
            Self::NonFinite { index } => {
                write!(
                    f,
                    "value at {index} is not finite or its scale overflows f16"
                )
            }
            Self::Awq(e) => write!(f, "{e}"),
            Self::Measure(e) => write!(f, "{e:?}"),
        }
    }
}

impl std::error::Error for CalibrateError {}

/// Activations a layer saw on sample inputs: `n` rows of `ic` finite f32 values.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSet {
    ic: usize,
    n: usize,
    x: Vec<f32>,
}

impl CalibrationSet {
    /// `x` is `[n, ic]` row-major; `n >= 1` and every value finite.
    pub fn new(ic: usize, x: Vec<f32>) -> Result<Self, CalibrateError> {
        if ic == 0 {
            return Err(CalibrateError::BadDim {
                what: "ic",
                value: 0,
            });
        }
        if x.is_empty() || !x.len().is_multiple_of(ic) {
            return Err(CalibrateError::BadLen {
                what: "calibration activations (a multiple of ic, at least one row)",
                got: x.len(),
                want: ic,
            });
        }
        if let Some(index) = x.iter().position(|v| !v.is_finite()) {
            return Err(CalibrateError::NonFinite { index });
        }
        Ok(Self {
            ic,
            n: x.len() / ic,
            x,
        })
    }

    /// Input features per sample.
    #[must_use]
    pub fn ic(&self) -> usize {
        self.ic
    }

    /// Number of samples.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The samples, `[n, ic]` row-major.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.x
    }

    /// `X W^T` as `[n, oc]`, accumulated in f64, for `w` `[oc, ic]` row-major.
    fn project(&self, w: &[f32], oc: usize) -> Vec<f32> {
        let mut y = Vec::with_capacity(self.n * oc);
        for row in self.x.chunks_exact(self.ic) {
            for wo in w.chunks_exact(self.ic) {
                let acc: f64 = row
                    .iter()
                    .zip(wo)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                y.push(acc as f32);
            }
        }
        y
    }
}

/// A calibrated layer with its error reports.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibrated {
    /// The quantized layer.
    pub awq: LogicalAwq,
    /// Dequantized weights against the input weights, with the proven bound.
    pub report: ErrorReport,
    /// `X W^T` against `X W'^T`, measured only; `None` when no [`CalibrationSet`] was given.
    pub output_report: Option<ErrorReport>,
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

/// `[IC, OC]` dequantized values to `[OC, IC]`.
fn to_out_in(deq: &[f32], ic: usize, oc: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; ic * oc];
    for i in 0..ic {
        for o in 0..oc {
            w[o * ic + i] = deq[i * oc + o];
        }
    }
    w
}

/// Checks shapes; `ic` and `oc` must be multiples of 8 (the AWQ packing) and `ic` of `group_size`.
fn validate(w: &[f32], oc: usize, ic: usize, group_size: usize) -> Result<(), CalibrateError> {
    if group_size == 0 {
        return Err(CalibrateError::BadGroupSize(0));
    }
    for (what, value) in [("oc", oc), ("ic", ic)] {
        if value == 0 {
            return Err(CalibrateError::BadDim { what, value });
        }
    }
    for value in [oc, ic] {
        if !value.is_multiple_of(8) {
            return Err(CalibrateError::BadDim {
                what: "oc and ic must each be a multiple of 8 (the AWQ packing)",
                value,
            });
        }
    }
    if !ic.is_multiple_of(group_size) {
        return Err(CalibrateError::BadDim {
            what: "ic (not a multiple of group_size)",
            value: ic,
        });
    }
    if w.len() != oc * ic {
        return Err(CalibrateError::BadLen {
            what: "weights",
            got: w.len(),
            want: oc * ic,
        });
    }
    Ok(())
}

/// Round-to-nearest quantization of `w` (`[oc, ic]`), returning the layer and each group's error bound
/// (indexed `[group * oc + o]`, 0 for an all-zero group, which is exact).
fn quantize(
    w: &[f32],
    oc: usize,
    ic: usize,
    group_size: usize,
) -> Result<(LogicalAwq, Vec<f64>), CalibrateError> {
    let groups = ic / group_size;
    let mut q = vec![0u8; ic * oc];
    let mut z = vec![0u8; groups * oc];
    let mut scales = vec![f16::from_f32(1.0); groups * oc];
    let mut bounds = vec![0.0f64; groups * oc];
    for o in 0..oc {
        for g in 0..groups {
            let start = o * ic + g * group_size;
            let run = &w[start..start + group_size];
            if let Some(i) = run.iter().position(|v| !v.is_finite()) {
                return Err(CalibrateError::NonFinite { index: start + i });
            }
            let lo = run.iter().fold(0.0f32, |m, &v| m.min(v));
            let hi = run.iter().fold(0.0f32, |m, &v| m.max(v));
            let range = hi - lo;
            if range == 0.0 {
                continue;
            }
            let s = f16_round_up(range / 15.0).ok_or(CalibrateError::NonFinite { index: start })?;
            let sf = s.to_f32();
            let zp = (-lo / sf).round().clamp(0.0, 15.0);
            z[g * oc + o] = zp as u8;
            scales[g * oc + o] = s;
            for (k, &v) in run.iter().enumerate() {
                q[(g * group_size + k) * oc + o] = ((v / sf).round() + zp).clamp(0.0, 15.0) as u8;
            }
            let sd = f64::from(sf);
            bounds[g * oc + o] =
                sd * (0.5 + 15.0 * 2f64.powi(-11) + 2f64.powi(-20)) + 2f64.powi(-25);
        }
    }
    let awq = LogicalAwq {
        ic,
        oc,
        group_size,
        q,
        z,
        scales,
    };
    Ok((awq, bounds))
}

fn check_calibration(c: &CalibrationSet, ic: usize) -> Result<(), CalibrateError> {
    if c.ic != ic {
        return Err(CalibrateError::BadLen {
            what: "calibration ic against the layer's ic",
            got: c.ic,
            want: ic,
        });
    }
    Ok(())
}

/// Round-to-nearest quantization of `w` (`[oc, ic]` row-major) into the AWQ layout, groups of `group_size`.
///
/// With `calibration` the result also carries the output error on those activations. `ic` must be a
/// multiple of `group_size`, and `calibration.ic()` must equal `ic`.
pub fn rtn_awq(
    w: &[f32],
    oc: usize,
    ic: usize,
    group_size: usize,
    calibration: Option<&CalibrationSet>,
) -> Result<Calibrated, CalibrateError> {
    validate(w, oc, ic, group_size)?;
    if let Some(c) = calibration {
        check_calibration(c, ic)?;
    }
    let (awq, bounds) = quantize(w, oc, ic, group_size)?;
    let worst = bounds.iter().fold(0.0f64, |m, &b| m.max(b));
    let decoded = to_out_in(
        &dequantize_logical(&awq).map_err(CalibrateError::Awq)?,
        ic,
        oc,
    );
    let report = ErrorReport::measure(w, &decoded)
        .map_err(CalibrateError::Measure)?
        .with_bound(worst);
    let output_report = calibration
        .map(|c| ErrorReport::measure(&c.project(w, oc), &c.project(&decoded, oc)))
        .transpose()
        .map_err(CalibrateError::Measure)?;
    Ok(Calibrated {
        awq,
        report,
        output_report,
    })
}

/// One grid point of the search.
struct Candidate {
    err: f64,
    s: Vec<f32>,
    awq: LogicalAwq,
    bounds: Vec<f64>,
    eff: Vec<f32>,
    alpha: f32,
}

/// Settings for [`search_awq`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqSearch {
    /// Input features per quantization group.
    pub group_size: usize,
    /// Number of exponents tried: `alpha = k / grid` for `k = 0..grid` (the published method uses 20).
    pub grid: usize,
}

impl Default for AwqSearch {
    fn default() -> Self {
        Self {
            group_size: 128,
            grid: 20,
        }
    }
}

/// The result of [`search_awq`].
///
/// The layer computes `x W^T` as `(x / s) (W s)^T`: [`awq`](Self::awq) quantizes `W s` (each input column `i`
/// scaled by `channel_scales[i]`), so the caller must divide input channel `i` of the activations by
/// `channel_scales[i]`, normally by folding it into the preceding op. The effective weights, which both
/// reports are measured on, are the dequantized values divided by `channel_scales[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Searched {
    /// The quantized scaled weights.
    pub awq: LogicalAwq,
    /// `s`, length `ic`; all 1.0 when `alpha` is 0.
    pub channel_scales: Vec<f32>,
    /// The exponent chosen: `s_i = mean|x_i|^alpha` (normalized).
    pub alpha: f32,
    /// Effective weights against the input weights, with a proven bound (see the crate docs).
    pub report: ErrorReport,
    /// `X W^T` against `X W_eff^T` for the chosen result; measured only.
    pub output_report: ErrorReport,
    /// The same for round-to-nearest (`alpha` 0), so the gain is stated, not assumed.
    pub baseline_output_report: ErrorReport,
}

/// AWQ-style scale search: per input channel `s_i = mean|x_i|^alpha`, normalized by `sqrt(max s * min s)`,
/// for each `alpha` on the grid quantize `W s` by round to nearest and keep the exponent whose effective
/// weights give the least squared output error on `calibration`. Ties go to the smaller `alpha`, and `alpha`
/// 0 is round to nearest, so the result is never worse than [`rtn_awq`] on `calibration`.
///
/// This re-derives the method of the AWQ paper (arXiv 2306.00978); no AutoAWQ code is used and the scales
/// are not checked against AutoAWQ's. A channel whose mean activation is below `1e-4` is treated as `1e-4`.
pub fn search_awq(
    w: &[f32],
    oc: usize,
    ic: usize,
    calibration: &CalibrationSet,
    settings: &AwqSearch,
) -> Result<Searched, CalibrateError> {
    validate(w, oc, ic, settings.group_size)?;
    check_calibration(calibration, ic)?;
    if settings.grid == 0 {
        return Err(CalibrateError::BadDim {
            what: "grid",
            value: 0,
        });
    }
    let gs = settings.group_size;
    let mean_abs: Vec<f64> = (0..ic)
        .map(|i| {
            let sum: f64 = calibration
                .x
                .chunks_exact(ic)
                .map(|row| f64::from(row[i]).abs())
                .sum();
            (sum / calibration.n as f64).max(1e-4)
        })
        .collect();
    let reference = calibration.project(w, oc);
    let sse = |y: &[f32]| -> f64 {
        reference
            .iter()
            .zip(y)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum()
    };
    let mut best: Option<Candidate> = None;
    let mut baseline: Option<Vec<f32>> = None;
    for k in 0..settings.grid {
        let alpha = k as f64 / settings.grid as f64;
        let raw: Vec<f64> = mean_abs.iter().map(|m| m.powf(alpha)).collect();
        let norm = (raw.iter().fold(0.0f64, |m, &v| m.max(v))
            * raw.iter().fold(f64::INFINITY, |m, &v| m.min(v)))
        .sqrt();
        let s: Vec<f32> = raw.iter().map(|&v| (v / norm) as f32).collect();
        let mut scaled = w.to_vec();
        for row in scaled.chunks_exact_mut(ic) {
            for (v, &si) in row.iter_mut().zip(&s) {
                *v *= si;
            }
        }
        let (awq, bounds) = quantize(&scaled, oc, ic, gs)?;
        let mut eff = to_out_in(
            &dequantize_logical(&awq).map_err(CalibrateError::Awq)?,
            ic,
            oc,
        );
        for row in eff.chunks_exact_mut(ic) {
            for (v, &si) in row.iter_mut().zip(&s) {
                *v /= si;
            }
        }
        let y = calibration.project(&eff, oc);
        let err = sse(&y);
        if k == 0 {
            baseline = Some(y);
        }
        if best.as_ref().is_none_or(|b| err < b.err) {
            best = Some(Candidate {
                err,
                s,
                awq,
                bounds,
                eff,
                alpha: alpha as f32,
            });
        }
    }
    let Candidate {
        s,
        awq,
        bounds,
        eff,
        alpha,
        ..
    } = best.expect("grid is at least 1");
    let groups = ic / gs;
    let mut worst = 0.0f64;
    for g in 0..groups {
        let smin = s[g * gs..(g + 1) * gs]
            .iter()
            .fold(f32::INFINITY, |m, &v| m.min(v));
        for o in 0..oc {
            worst = worst.max(bounds[g * oc + o] / f64::from(smin));
        }
    }
    // Dividing by `s_i` in f32 rounds each value by at most 2^-24 of its size.
    let rounding = 2f64.powi(-23) * eff.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let report = ErrorReport::measure(w, &eff)
        .map_err(CalibrateError::Measure)?
        .with_bound(worst * (1.0 + 2f64.powi(-22)) + rounding);
    let output_report = ErrorReport::measure(&reference, &calibration.project(&eff, oc))
        .map_err(CalibrateError::Measure)?;
    let baseline_output_report = ErrorReport::measure(&reference, &baseline.expect("k = 0 ran"))
        .map_err(CalibrateError::Measure)?;
    Ok(Searched {
        awq,
        channel_scales: s,
        alpha,
        report,
        output_report,
        baseline_output_report,
    })
}

#[cfg(test)]
mod tests;
