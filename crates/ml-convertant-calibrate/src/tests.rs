use super::*;
use ml_convertant_awq::dequantize_logical;
use ml_convertant_lora::{LoraAdapter, merge_lora_into_awq};

/// Deterministic pseudo-random f32 in about [-2, 2).
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// Eight identical output rows, each the four values twice (ic = 8).
fn rows(four: &[f32; 4]) -> Vec<f32> {
    let row: Vec<f32> = four.iter().chain(four).copied().collect();
    (0..8).flat_map(|_| row.clone()).collect()
}

/// The codes of output feature 0 across the input features.
fn column0(l: &LogicalAwq) -> Vec<u8> {
    (0..l.ic).map(|i| l.q[i * l.oc]).collect()
}

#[test]
fn hand_worked_positive_group() {
    // range 4.5, scale = f16 round-up of 0.3 = 0.30004883, zero 0, codes round(v / scale).
    let r = rtn_awq(&rows(&[0.0, 1.5, 3.0, 4.5]), 8, 8, 8, None).unwrap();
    assert_eq!(column0(&r.awq), vec![0, 5, 10, 15, 0, 5, 10, 15]);
    assert_eq!(r.awq.z[0], 0);
    assert_eq!(r.awq.scales[0].to_f32(), 0.300_048_83);
    assert!(r.output_report.is_none());
}

#[test]
fn hand_worked_signed_group() {
    // range 2, scale = f16 round-up of 2/15 = 0.13342285, so -min/scale = 7.495 and the zero is 7.
    let r = rtn_awq(&rows(&[-1.0, 0.0, 0.0, 1.0]), 8, 8, 8, None).unwrap();
    assert_eq!(r.awq.z[0], 7);
    assert_eq!(column0(&r.awq), vec![0, 7, 7, 14, 0, 7, 7, 14]);
}

#[test]
fn an_all_positive_group_still_spans_zero() {
    // The range is widened to include 0: lo = 0, hi = 4, scale = f16 round-up of 4/15 = 0.26684570, zero 0.
    let r = rtn_awq(&rows(&[1.0, 2.0, 3.0, 4.0]), 8, 8, 8, None).unwrap();
    assert_eq!(r.awq.z[0], 0);
    assert_eq!(column0(&r.awq), vec![4, 7, 11, 15, 4, 7, 11, 15]);
}

#[test]
fn all_zero_group_is_exact() {
    let r = rtn_awq(&[0.0; 64], 8, 8, 4, None).unwrap();
    assert_eq!(
        (r.awq.q.iter().max(), r.awq.z.iter().max()),
        (Some(&0), Some(&0))
    );
    assert!(r.awq.scales.iter().all(|s| s.to_f32() == 1.0));
    assert!(r.report.is_clean());
}

#[test]
fn error_is_within_the_proven_bound() {
    let (oc, ic, gs) = (16, 64, 32);
    let w = noise(oc * ic, 7);
    let r = rtn_awq(&w, oc, ic, gs, None).unwrap();
    assert_eq!(r.report.within_bound(), Some(true));
    assert!(
        r.report.max_abs > 0.0,
        "a nonzero error is expected for random weights"
    );
}

#[test]
fn matches_the_lora_requantizer_bit_for_bit() {
    // merge_lora_into_awq with a zero adapter requantizes the dequantized base. Quantizing that same
    // dequantized matrix here must give the identical layer.
    let (oc, ic, gs) = (16, 64, 32);
    let base = rtn_awq(&noise(oc * ic, 11), oc, ic, gs, None).unwrap().awq;
    let deq = dequantize_logical(&base).unwrap();
    let mut w = vec![0.0f32; oc * ic];
    for i in 0..ic {
        for o in 0..oc {
            w[o * ic + i] = deq[i * oc + o];
        }
    }
    let zero = LoraAdapter {
        out_features: oc,
        in_features: ic,
        rank: 1,
        a: noise(ic, 3),
        b: vec![0.0; oc],
        scale: 1.0,
    };
    let theirs = merge_lora_into_awq(&base, &zero).unwrap().output;
    let ours = rtn_awq(&w, oc, ic, gs, None).unwrap().awq;
    assert_eq!(ours, theirs);
}

#[test]
fn output_report_is_present_and_measures_the_projection() {
    let (oc, ic, gs) = (8, 32, 16);
    let w = noise(oc * ic, 5);
    let x = CalibrationSet::new(ic, noise(4 * ic, 9)).unwrap();
    assert_eq!(x.n(), 4);
    let r = rtn_awq(&w, oc, ic, gs, Some(&x)).unwrap();
    let out = r.output_report.expect("calibration given");
    assert_eq!(out.n, 4 * oc);
    assert!(out.max_abs > 0.0);
    assert_eq!(
        out.within_bound(),
        None,
        "output error is measured, not bounded"
    );
}

#[test]
fn output_error_is_zero_when_weights_are_exactly_representable() {
    // Weights that RTN reproduces exactly: dequantize a quantized layer and quantize that again.
    let (oc, ic, gs) = (8, 32, 16);
    let base = rtn_awq(&noise(oc * ic, 21), oc, ic, gs, None).unwrap().awq;
    let deq = dequantize_logical(&base).unwrap();
    let mut w = vec![0.0f32; oc * ic];
    for i in 0..ic {
        for o in 0..oc {
            w[o * ic + i] = deq[i * oc + o];
        }
    }
    let x = CalibrationSet::new(ic, noise(3 * ic, 2)).unwrap();
    let r = rtn_awq(&w, oc, ic, gs, Some(&x)).unwrap();
    assert!(r.report.is_clean());
    assert!(r.output_report.unwrap().is_clean());
}

#[test]
fn bad_inputs_are_refused() {
    let w = vec![1.0f32; 64];
    assert_eq!(
        rtn_awq(&w, 8, 8, 0, None).unwrap_err(),
        CalibrateError::BadGroupSize(0)
    );
    assert!(matches!(
        rtn_awq(&w, 8, 8, 3, None),
        Err(CalibrateError::BadDim { .. })
    ));
    assert!(matches!(
        rtn_awq(&w, 4, 8, 4, None),
        Err(CalibrateError::BadDim { .. })
    ));
    assert!(matches!(
        rtn_awq(&w, 8, 16, 4, None),
        Err(CalibrateError::BadLen { .. })
    ));
    let mut bad = w.clone();
    bad[5] = f32::NAN;
    assert_eq!(
        rtn_awq(&bad, 8, 8, 4, None).unwrap_err(),
        CalibrateError::NonFinite { index: 5 }
    );
    let x = CalibrationSet::new(4, vec![0.0; 4]).unwrap();
    assert!(matches!(
        rtn_awq(&w, 8, 8, 4, Some(&x)),
        Err(CalibrateError::BadLen { .. })
    ));
    assert!(CalibrationSet::new(0, vec![1.0]).is_err());
    assert!(CalibrationSet::new(4, vec![]).is_err());
    assert!(CalibrationSet::new(4, vec![1.0; 6]).is_err());
    assert_eq!(
        CalibrationSet::new(2, vec![1.0, f32::INFINITY]).unwrap_err(),
        CalibrateError::NonFinite { index: 1 }
    );
}

/// Activations with one very large channel and the rest around 1, which is where the search should help.
fn outlier_activations(n: usize, ic: usize, hot: usize) -> CalibrationSet {
    let mut x = noise(n * ic, 17);
    for row in x.chunks_exact_mut(ic) {
        row[hot] *= 40.0;
    }
    CalibrationSet::new(ic, x).unwrap()
}

/// `X W^T` measured independently of the crate: effective weights from the layer and its channel scales.
fn effective_weights(s: &Searched) -> Vec<f32> {
    let deq = dequantize_logical(&s.awq).unwrap();
    let (ic, oc) = (s.awq.ic, s.awq.oc);
    let mut w = vec![0.0f32; oc * ic];
    for i in 0..ic {
        for o in 0..oc {
            w[o * ic + i] = deq[i * oc + o] / s.channel_scales[i];
        }
    }
    w
}

#[test]
fn search_beats_rtn_when_one_channel_dominates() {
    let (oc, ic, gs) = (16, 64, 32);
    let w = noise(oc * ic, 31);
    let x = outlier_activations(32, ic, 5);
    let s = search_awq(
        &w,
        oc,
        ic,
        &x,
        &AwqSearch {
            group_size: gs,
            grid: 20,
        },
    )
    .unwrap();
    assert!(s.alpha > 0.0, "alpha = {}", s.alpha);
    assert!(
        s.output_report.rmse < s.baseline_output_report.rmse,
        "{} vs {}",
        s.output_report.rmse,
        s.baseline_output_report.rmse
    );
}

#[test]
fn search_is_never_worse_than_rtn() {
    let (oc, ic, gs) = (8, 32, 16);
    for seed in 1..6u32 {
        let w = noise(oc * ic, seed);
        let x = CalibrationSet::new(ic, noise(10 * ic, seed + 100)).unwrap();
        let s = search_awq(
            &w,
            oc,
            ic,
            &x,
            &AwqSearch {
                group_size: gs,
                grid: 20,
            },
        )
        .unwrap();
        let rtn = rtn_awq(&w, oc, ic, gs, Some(&x)).unwrap();
        assert_eq!(s.baseline_output_report, rtn.output_report.unwrap());
        assert!(s.output_report.rmse <= s.baseline_output_report.rmse);
    }
}

#[test]
fn a_grid_of_one_is_round_to_nearest() {
    let (oc, ic, gs) = (8, 32, 16);
    let w = noise(oc * ic, 4);
    let x = outlier_activations(8, ic, 2);
    let s = search_awq(
        &w,
        oc,
        ic,
        &x,
        &AwqSearch {
            group_size: gs,
            grid: 1,
        },
    )
    .unwrap();
    let rtn = rtn_awq(&w, oc, ic, gs, Some(&x)).unwrap();
    assert_eq!(s.alpha, 0.0);
    assert!(s.channel_scales.iter().all(|&v| v == 1.0));
    assert_eq!(s.awq, rtn.awq);
    // Same measured error; the search's bound adds slack for dividing by the channel scales, so it is looser.
    assert_eq!(
        (s.report.max_abs, s.report.rmse, s.report.mean_abs),
        (rtn.report.max_abs, rtn.report.rmse, rtn.report.mean_abs)
    );
    assert!(s.report.bound >= rtn.report.bound);
}

#[test]
fn search_reports_match_an_independent_measurement_and_the_bound_holds() {
    let (oc, ic, gs) = (16, 64, 32);
    let w = noise(oc * ic, 8);
    let x = outlier_activations(16, ic, 9);
    let s = search_awq(
        &w,
        oc,
        ic,
        &x,
        &AwqSearch {
            group_size: gs,
            grid: 20,
        },
    )
    .unwrap();
    let eff = effective_weights(&s);
    assert_eq!(s.report.within_bound(), Some(true));
    assert_eq!(
        ErrorReport::measure(&w, &eff).unwrap().max_abs,
        s.report.max_abs
    );
    // (x / s)(W s)^T against x W^T, computed here without the crate's projection.
    let deq = dequantize_logical(&s.awq).unwrap();
    let mut worst = 0.0f64;
    for row in x.samples().chunks_exact(ic) {
        for o in 0..oc {
            let exact: f64 = (0..ic)
                .map(|i| f64::from(row[i]) * f64::from(w[o * ic + i]))
                .sum();
            let got: f64 = (0..ic)
                .map(|i| f64::from(row[i] / s.channel_scales[i]) * f64::from(deq[i * oc + o]))
                .sum();
            worst = worst.max((exact - got).abs());
        }
    }
    assert!(
        (worst - s.output_report.max_abs).abs() < 1e-3 * (1.0 + worst),
        "{worst} vs {}",
        s.output_report.max_abs
    );
}

#[test]
fn search_is_deterministic() {
    let (oc, ic, gs) = (8, 32, 16);
    let w = noise(oc * ic, 6);
    let x = outlier_activations(8, ic, 1);
    let cfg = AwqSearch {
        group_size: gs,
        grid: 20,
    };
    assert_eq!(
        search_awq(&w, oc, ic, &x, &cfg),
        search_awq(&w, oc, ic, &x, &cfg)
    );
}

#[test]
fn search_refuses_bad_inputs() {
    let w = noise(64, 1);
    let x = CalibrationSet::new(8, noise(16, 2)).unwrap();
    let ok = AwqSearch {
        group_size: 8,
        grid: 4,
    };
    assert!(search_awq(&w, 8, 8, &x, &ok).is_ok());
    assert!(matches!(
        search_awq(
            &w,
            8,
            8,
            &x,
            &AwqSearch {
                grid: 0,
                ..ok.clone()
            }
        ),
        Err(CalibrateError::BadDim { .. })
    ));
    assert!(matches!(
        search_awq(
            &w,
            8,
            8,
            &CalibrationSet::new(4, vec![1.0; 8]).unwrap(),
            &ok
        ),
        Err(CalibrateError::BadLen { .. })
    ));
    assert_eq!(
        search_awq(
            &w,
            8,
            8,
            &x,
            &AwqSearch {
                group_size: 0,
                grid: 4
            }
        )
        .unwrap_err(),
        CalibrateError::BadGroupSize(0)
    );
}

#[test]
fn ties_go_to_the_smaller_alpha() {
    // Zero weights give zero error at every alpha, so the first (alpha 0) must win.
    let x = outlier_activations(4, 16, 3);
    let s = search_awq(
        &[0.0; 128],
        8,
        16,
        &x,
        &AwqSearch {
            group_size: 8,
            grid: 20,
        },
    )
    .unwrap();
    assert_eq!(s.alpha, 0.0);
    assert!(s.channel_scales.iter().all(|&v| v == 1.0));
}
