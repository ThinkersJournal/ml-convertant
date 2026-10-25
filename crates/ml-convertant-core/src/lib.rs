//! The error-measurement contract and the [`Conversion`] trait.
//!
//! Every lossy conversion returns its output together with an [`ErrorReport`]; a conversion that
//! cannot report does not implement [`Conversion`].

use std::fmt;

/// Names the reference an output was compared byte-for-byte against, e.g. `"llama.cpp@9d57ce456c"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceId(pub &'static str);

/// Source and converted slices had different lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch {
    /// Length of the source slice.
    pub source: usize,
    /// Length of the converted slice.
    pub converted: usize,
}

impl fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "source has {} elements but converted has {}",
            self.source, self.converted
        )
    }
}

impl std::error::Error for LengthMismatch {}

/// Element-wise error of a converted tensor against its source values.
///
/// Measured in `f64` over `(x, x')` pairs where both values are finite. Non-finite values are
/// never folded into the statistics and never hidden: they are counted, and a pair whose
/// non-finite state changed is counted in [`finiteness_changed`](Self::finiteness_changed), which
/// makes [`is_clean`](Self::is_clean) false.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorReport {
    /// Total elements compared (length of both slices).
    pub n: usize,
    /// Pairs where both values were finite; the denominator of `mean_abs` and `rmse`.
    pub n_measured: usize,
    /// max |x - x'| over measured pairs (0.0 when none).
    pub max_abs: f64,
    /// Index of the first element reaching `max_abs`; `None` when nothing was measured.
    pub max_abs_index: Option<usize>,
    /// mean |x - x'| over measured pairs (0.0 when none).
    pub mean_abs: f64,
    /// sqrt(mean (x - x')^2) over measured pairs (0.0 when none).
    pub rmse: f64,
    /// Non-finite values in the source.
    pub nonfinite_in: usize,
    /// Non-finite values in the converted output.
    pub nonfinite_out: usize,
    /// Pairs where exactly one side was non-finite, or both were non-finite but different
    /// (NaN vs. Inf, or +Inf vs. -Inf).
    pub finiteness_changed: usize,
    /// The format's proven worst-case |x - x'| for this input, when known.
    pub bound: Option<f64>,
    /// Set only when the output bytes were compared equal to this named reference.
    pub bit_exact_vs: Option<ReferenceId>,
}

impl ErrorReport {
    /// Measures `converted` against `source`.
    pub fn measure(source: &[f32], converted: &[f32]) -> Result<Self, LengthMismatch> {
        if source.len() != converted.len() {
            return Err(LengthMismatch {
                source: source.len(),
                converted: converted.len(),
            });
        }
        let mut r = Self {
            n: source.len(),
            n_measured: 0,
            max_abs: 0.0,
            max_abs_index: None,
            mean_abs: 0.0,
            rmse: 0.0,
            nonfinite_in: 0,
            nonfinite_out: 0,
            finiteness_changed: 0,
            bound: None,
            bit_exact_vs: None,
        };
        let (mut sum_abs, mut sum_sq) = (0.0f64, 0.0f64);
        for (i, (&x, &y)) in source.iter().zip(converted).enumerate() {
            let (xf, yf) = (x.is_finite(), y.is_finite());
            r.nonfinite_in += usize::from(!xf);
            r.nonfinite_out += usize::from(!yf);
            if xf && yf {
                let d = (f64::from(x) - f64::from(y)).abs();
                r.n_measured += 1;
                sum_abs += d;
                sum_sq += d * d;
                if r.max_abs_index.is_none() || d > r.max_abs {
                    r.max_abs = d;
                    r.max_abs_index = Some(i);
                }
            } else if xf != yf || !same_nonfinite(x, y) {
                r.finiteness_changed += 1;
            }
        }
        if r.n_measured > 0 {
            let m = r.n_measured as f64;
            r.mean_abs = sum_abs / m;
            r.rmse = (sum_sq / m).sqrt();
        }
        Ok(r)
    }

    /// Records the format's proven worst-case error.
    #[must_use]
    pub fn with_bound(mut self, bound: f64) -> Self {
        self.bound = Some(bound);
        self
    }

    /// Records that the output was compared byte-for-byte equal to `reference`.
    #[must_use]
    pub fn with_reference(mut self, reference: ReferenceId) -> Self {
        self.bit_exact_vs = Some(reference);
        self
    }

    /// `true` when no pair changed its non-finite state.
    pub fn is_clean(&self) -> bool {
        self.finiteness_changed == 0
    }

    /// `Some(max_abs <= bound)` when a bound is recorded; `None` otherwise.
    pub fn within_bound(&self) -> Option<bool> {
        self.bound.map(|b| self.max_abs <= b)
    }
}

/// Both non-finite and the same kind: NaN with NaN, or Inf of equal sign.
fn same_nonfinite(x: f32, y: f32) -> bool {
    (x.is_nan() && y.is_nan()) || x == y
}

/// A conversion's output together with its measured error.
#[derive(Debug, Clone, PartialEq)]
pub struct Converted<T> {
    /// The converted data.
    pub output: T,
    /// Error of `output` against the input; see [`ErrorReport`].
    pub report: ErrorReport,
}

/// A data conversion that reports its own error.
pub trait Conversion {
    /// Input data.
    type Input: ?Sized;
    /// Output data.
    type Output;
    /// Failure type.
    type Error;

    /// Converts `input`, returning the output and its [`ErrorReport`].
    fn convert(&self, input: &Self::Input) -> Result<Converted<Self::Output>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_zero_error() {
        let v = [1.0f32, -2.5, 0.0];
        let r = ErrorReport::measure(&v, &v).unwrap();
        assert_eq!((r.n, r.n_measured), (3, 3));
        assert_eq!((r.max_abs, r.mean_abs, r.rmse), (0.0, 0.0, 0.0));
        assert!(r.is_clean());
    }

    #[test]
    fn known_values() {
        let r = ErrorReport::measure(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.5, 3.0, 2.0]).unwrap();
        assert_eq!(r.max_abs, 2.0);
        assert_eq!(r.max_abs_index, Some(3));
        assert_eq!(r.mean_abs, 0.625);
        assert_eq!(r.rmse, (4.25f64 / 4.0).sqrt());
    }

    #[test]
    fn first_index_wins_ties() {
        let r = ErrorReport::measure(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
        assert_eq!(r.max_abs_index, Some(0));
    }

    #[test]
    fn nonfinite_is_counted_not_measured() {
        let r = ErrorReport::measure(
            &[1.0, f32::NAN, f32::INFINITY, 4.0],
            &[1.5, f32::NAN, 7.0, 4.0],
        )
        .unwrap();
        assert_eq!((r.nonfinite_in, r.nonfinite_out), (2, 1));
        assert_eq!(r.n_measured, 2);
        // Inf -> 7.0 changed; NaN -> NaN is preserved.
        assert_eq!(r.finiteness_changed, 1);
        assert!(!r.is_clean());
        assert_eq!(r.max_abs, 0.5);
    }

    #[test]
    fn preserved_nonfinite_is_clean() {
        let v = [f32::INFINITY, f32::NEG_INFINITY];
        assert!(ErrorReport::measure(&v, &v).unwrap().is_clean());
        let flipped = ErrorReport::measure(&[f32::INFINITY], &[f32::NEG_INFINITY]).unwrap();
        assert_eq!(flipped.finiteness_changed, 1);
    }

    #[test]
    fn empty_and_all_nonfinite() {
        let e = ErrorReport::measure(&[], &[]).unwrap();
        assert_eq!((e.n, e.max_abs_index, e.mean_abs), (0, None, 0.0));
        let n = ErrorReport::measure(&[f32::NAN], &[f32::NAN]).unwrap();
        assert_eq!((n.n_measured, n.max_abs_index), (0, None));
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let e = ErrorReport::measure(&[1.0], &[]).unwrap_err();
        assert_eq!(
            e,
            LengthMismatch {
                source: 1,
                converted: 0
            }
        );
    }

    #[test]
    fn bound_and_reference() {
        let r = ErrorReport::measure(&[1.0], &[1.25]).unwrap();
        assert_eq!(r.within_bound(), None);
        assert_eq!(r.clone().with_bound(0.25).within_bound(), Some(true));
        assert_eq!(r.clone().with_bound(0.2).within_bound(), Some(false));
        assert_eq!(
            r.with_reference(ReferenceId("ref")).bit_exact_vs,
            Some(ReferenceId("ref"))
        );
    }

    struct Halve;
    impl Conversion for Halve {
        type Input = [f32];
        type Output = Vec<f32>;
        type Error = LengthMismatch;
        fn convert(&self, input: &[f32]) -> Result<Converted<Vec<f32>>, LengthMismatch> {
            let output: Vec<f32> = input.iter().map(|x| x / 2.0).collect();
            let report = ErrorReport::measure(input, &output)?;
            Ok(Converted { output, report })
        }
    }

    #[test]
    fn trait_returns_report() {
        let c = Halve.convert(&[2.0, 4.0]).unwrap();
        assert_eq!(c.output, vec![1.0, 2.0]);
        assert_eq!(c.report.max_abs, 2.0);
    }
}
