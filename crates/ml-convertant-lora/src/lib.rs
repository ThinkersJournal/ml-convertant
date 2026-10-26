//! Offline LoRA merge with measured error.
//!
//! # Operation
//!
//! For a base weight `W` of shape `[out, in]` and an adapter with `A [rank, in]`, `B [out, rank]`
//! and a scale (`alpha / rank`, or `alpha / sqrt(rank)` for rank-stabilized LoRA):
//!
//! ```text
//! W'[o, i] = W[o, i] + scale * sum_k B[o, k] * A[k, i]
//! ```
//!
//! as PEFT's `get_delta_weight` / `merge` compute it. The sum runs over `k` ascending in f32,
//! starting from zero, with no fused multiply-add, then is multiplied by `scale` and added to `W`;
//! the result is therefore deterministic across platforms. Weights stored transposed
//! (`fan_in_fan_out`) must be transposed by the caller first.
//!
//! # Error
//!
//! [`ErrorReport`] compares the f32 result with the same merge evaluated in f64 and rounded to f32.
//! With `u = 2^-24`, `gamma(n) = n * u / (1 - n * u)` and `S = sum_k |B[o, k] * A[k, i]|`, each
//! element satisfies
//!
//! ```text
//! |W'_f32 - W'_exact| <= gamma(rank + 4) * (|W| + |scale| * S)
//! ```
//!
//! (the dot product contributes `gamma(rank)`, the scale product, the rounding of `scale` itself and
//! the final addition the rest), and the f64 reference rounded to f32 is within `u * |W'_exact|` of
//! the exact value, which the recorded bound adds. [`lora_merge_bound`] computes the maximum over
//! all elements and it is recorded as [`ErrorReport::bound`]. Non-finite inputs are not errors; the
//! report counts them.

use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

/// Why a LoRA merge failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    /// A dimension (`out`, `in` or `rank`) was zero.
    ZeroDim(&'static str),
    /// A buffer has the wrong length.
    BadLen {
        /// Which buffer.
        what: &'static str,
        /// Its length.
        got: usize,
        /// The expected length.
        want: usize,
    },
    /// Measuring failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl fmt::Display for LoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDim(w) => write!(f, "LoRA {w} must be non-zero"),
            Self::BadLen { what, got, want } => {
                write!(f, "LoRA {what} has {got} elements, expected {want}")
            }
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for LoraError {}

/// `alpha / rank`, the standard LoRA scale, in f32.
pub fn lora_scale(alpha: f32, rank: usize) -> f32 {
    alpha / rank as f32
}

/// `alpha / sqrt(rank)`, the rank-stabilized LoRA scale, in f32.
pub fn rslora_scale(alpha: f32, rank: usize) -> f32 {
    alpha / (rank as f64).sqrt() as f32
}

/// A LoRA adapter for one linear layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapter {
    /// Output features.
    pub out_features: usize,
    /// Input features.
    pub in_features: usize,
    /// Adapter rank.
    pub rank: usize,
    /// `[rank, in_features]`, PEFT's `lora_A.weight`.
    pub a: Vec<f32>,
    /// `[out_features, rank]`, PEFT's `lora_B.weight`.
    pub b: Vec<f32>,
    /// Multiplier applied to `B @ A`.
    pub scale: f32,
}

impl LoraAdapter {
    fn validate(&self) -> Result<(), LoraError> {
        for (what, v) in [
            ("out_features", self.out_features),
            ("in_features", self.in_features),
            ("rank", self.rank),
        ] {
            if v == 0 {
                return Err(LoraError::ZeroDim(what));
            }
        }
        check_len("A", self.a.len(), self.rank * self.in_features)?;
        check_len("B", self.b.len(), self.out_features * self.rank)
    }
}

fn check_len(what: &'static str, got: usize, want: usize) -> Result<(), LoraError> {
    if got == want {
        Ok(())
    } else {
        Err(LoraError::BadLen { what, got, want })
    }
}

/// Merges `adapter` into `w` (`[out, in]` row-major) in f32.
pub fn merge_lora(w: &[f32], adapter: &LoraAdapter) -> Result<Vec<f32>, LoraError> {
    adapter.validate()?;
    let (out, inn, r) = (adapter.out_features, adapter.in_features, adapter.rank);
    check_len("W", w.len(), out * inn)?;
    let mut merged = Vec::with_capacity(w.len());
    let mut acc = vec![0.0f32; inn];
    for o in 0..out {
        acc.fill(0.0);
        for k in 0..r {
            let bk = adapter.b[o * r + k];
            for (acc_i, &a_ki) in acc.iter_mut().zip(&adapter.a[k * inn..(k + 1) * inn]) {
                *acc_i += bk * a_ki;
            }
        }
        merged.extend(
            w[o * inn..(o + 1) * inn]
                .iter()
                .zip(&acc)
                .map(|(&w_oi, &s)| w_oi + adapter.scale * s),
        );
    }
    Ok(merged)
}

/// The same merge evaluated in f64 and rounded to f32 once.
pub fn merge_lora_f64_reference(w: &[f32], adapter: &LoraAdapter) -> Result<Vec<f32>, LoraError> {
    adapter.validate()?;
    let (out, inn, r) = (adapter.out_features, adapter.in_features, adapter.rank);
    check_len("W", w.len(), out * inn)?;
    let scale = f64::from(adapter.scale);
    let mut merged = Vec::with_capacity(w.len());
    for o in 0..out {
        for i in 0..inn {
            let s: f64 = (0..r)
                .map(|k| f64::from(adapter.b[o * r + k]) * f64::from(adapter.a[k * inn + i]))
                .sum();
            merged.push((f64::from(w[o * inn + i]) + scale * s) as f32);
        }
    }
    Ok(merged)
}

/// The proven worst-case |W'_f32 - W'_reference| over all elements (see the crate docs).
pub fn lora_merge_bound(w: &[f32], adapter: &LoraAdapter) -> Result<f64, LoraError> {
    adapter.validate()?;
    let (out, inn, r) = (adapter.out_features, adapter.in_features, adapter.rank);
    check_len("W", w.len(), out * inn)?;
    let u = 2f64.powi(-24);
    let n = (r + 4) as f64;
    let gamma = n * u / (1.0 - n * u);
    let scale = f64::from(adapter.scale);
    let mut worst = 0.0f64;
    for o in 0..out {
        for i in 0..inn {
            let s: f64 = (0..r)
                .map(|k| {
                    (f64::from(adapter.b[o * r + k]) * f64::from(adapter.a[k * inn + i])).abs()
                })
                .sum();
            let wv = f64::from(w[o * inn + i]).abs();
            let exact_mag = wv + scale.abs() * s;
            worst = worst.max(gamma * exact_mag + u * exact_mag);
        }
    }
    Ok(worst)
}

/// Merge of a fixed adapter into weights, reporting the error against the f64 reference.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraMerge {
    /// The adapter to merge.
    pub adapter: LoraAdapter,
}

impl Conversion for LoraMerge {
    type Input = [f32];
    type Output = Vec<f32>;
    type Error = LoraError;

    fn convert(&self, w: &[f32]) -> Result<Converted<Vec<f32>>, LoraError> {
        let output = merge_lora(w, &self.adapter)?;
        let reference = merge_lora_f64_reference(w, &self.adapter)?;
        let bound = lora_merge_bound(w, &self.adapter)?;
        let report = ErrorReport::measure(&reference, &output)
            .map_err(LoraError::Measure)?
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

    fn adapter(out: usize, inn: usize, rank: usize, scale: f32, seed: u64) -> LoraAdapter {
        let mut s = seed;
        LoraAdapter {
            out_features: out,
            in_features: inn,
            rank,
            a: (0..rank * inn).map(|_| lcg(&mut s)).collect(),
            b: (0..out * rank).map(|_| lcg(&mut s)).collect(),
            scale,
        }
    }

    #[test]
    fn hand_computed_rank_one() {
        // B = [[1], [2]], A = [[3, 4, 5]], scale 0.5, W = 0:
        // B @ A = [[3, 4, 5], [6, 8, 10]] -> *0.5
        let ad = LoraAdapter {
            out_features: 2,
            in_features: 3,
            rank: 1,
            a: vec![3.0, 4.0, 5.0],
            b: vec![1.0, 2.0],
            scale: 0.5,
        };
        let m = merge_lora(&[0.0; 6], &ad).unwrap();
        assert_eq!(m, vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0]);
        let m = merge_lora(&[1.0; 6], &ad).unwrap();
        assert_eq!(m, vec![2.5, 3.0, 3.5, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn hand_computed_rank_two_sums_over_rank() {
        // B = [[1, 10]], A = [[1, 2], [3, 4]]: B @ A = [31, 42].
        let ad = LoraAdapter {
            out_features: 1,
            in_features: 2,
            rank: 2,
            a: vec![1.0, 2.0, 3.0, 4.0],
            b: vec![1.0, 10.0],
            scale: 1.0,
        };
        assert_eq!(merge_lora(&[0.0, 0.0], &ad).unwrap(), vec![31.0, 42.0]);
    }

    #[test]
    fn zero_b_leaves_weights_bit_identical() {
        let mut ad = adapter(4, 6, 2, 2.0, 1);
        ad.b.fill(0.0);
        let w: Vec<f32> = (0..24).map(|i| i as f32 * 0.37 - 3.0).collect();
        let c = LoraMerge { adapter: ad }.convert(&w).unwrap();
        assert_eq!(c.output, w);
        assert_eq!(c.report.max_abs, 0.0);
    }

    #[test]
    fn scales() {
        assert_eq!(lora_scale(16.0, 8), 2.0);
        assert_eq!(rslora_scale(8.0, 4), 4.0);
    }

    #[test]
    fn rejects_bad_geometry() {
        let ad = adapter(2, 3, 1, 1.0, 1);
        assert_eq!(
            merge_lora(&[0.0; 5], &ad),
            Err(LoraError::BadLen {
                what: "W",
                got: 5,
                want: 6
            })
        );
        let mut zero = ad.clone();
        zero.rank = 0;
        assert_eq!(
            merge_lora(&[0.0; 6], &zero),
            Err(LoraError::ZeroDim("rank"))
        );
        let mut short = ad;
        short.a.pop();
        assert!(matches!(
            merge_lora(&[0.0; 6], &short),
            Err(LoraError::BadLen { what: "A", .. })
        ));
    }

    #[test]
    fn bound_holds_on_random_adapters() {
        for round in 0..300usize {
            let (out, inn, rank) = (1 + round % 7, 1 + (round * 3) % 33, 1 + round % 16);
            let scale = [0.25f32, 1.0, 2.0, 16.0][round % 4];
            let ad = adapter(out, inn, rank, scale, round as u64 + 1);
            let mut s = 1000 + round as u64;
            let mag = 10f32.powi((round % 5) as i32 - 2);
            let w: Vec<f32> = (0..out * inn).map(|_| lcg(&mut s) * mag).collect();
            let c = LoraMerge { adapter: ad }.convert(&w).unwrap();
            assert_eq!(c.report.within_bound(), Some(true), "round {round}");
            assert!(c.report.is_clean());
        }
    }

    #[test]
    fn nonfinite_inputs_are_counted_not_hidden() {
        let ad = adapter(2, 2, 1, 1.0, 5);
        let w = [f32::NAN, 0.0, 0.0, 0.0];
        let c = LoraMerge { adapter: ad }.convert(&w).unwrap();
        // The report compares the f64 reference with the output; both carry the NaN.
        assert_eq!(c.report.nonfinite_in, 1);
        assert_eq!(c.report.nonfinite_out, 1);
        assert!(
            c.report.is_clean(),
            "NaN in both reference and output is preserved"
        );
    }
}
