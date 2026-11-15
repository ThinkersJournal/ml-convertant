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

/// A base whose every group decodes to exactly the integers `0..=15` times scale 1 (zero point 0):
/// the requantizer must recover it bit for bit when nothing is merged.
fn integer_base(ic: usize, oc: usize, gs: usize) -> LogicalAwq {
    let groups = ic / gs;
    let mut q = vec![0u8; ic * oc];
    for i in 0..ic {
        for o in 0..oc {
            q[i * oc + o] = ((i + 3 * o) % 16) as u8;
        }
    }
    // Make 15 appear in every group so the range is exactly 15 and the scale exactly 1.
    for g in 0..groups {
        for o in 0..oc {
            q[(g * gs) * oc + o] = 15;
        }
    }
    LogicalAwq {
        ic,
        oc,
        group_size: gs,
        q,
        z: vec![0; groups * oc],
        scales: vec![f16::from_f32(1.0); groups * oc],
    }
}

#[test]
fn nothing_merged_into_an_exactly_representable_base_changes_no_code_and_reports_zero_error() {
    let (ic, oc, gs) = (64, 8, 32);
    let base = integer_base(ic, oc, gs);
    let ad = adapter(oc, ic, 2, 1.0, 3).with_strength(0.0);
    let got = merge_lora_into_awq(&base, &ad).unwrap();
    assert_eq!(got.output, base);
    assert_eq!(got.report.max_abs, 0.0);
    assert_eq!(got.report.within_bound(), Some(true));
}

#[test]
fn the_chain_is_within_its_bound_and_actually_loses_something() {
    let mut seed = 1;
    for (ic, oc, gs, scale) in [(64, 8, 32, 0.5), (128, 16, 64, 1.0), (64, 8, 64, 4.0)] {
        seed += 1;
        let base = integer_base(ic, oc, gs);
        let ad = adapter(oc, ic, 3, scale, seed);
        let got = merge_lora_into_awq(&base, &ad).unwrap();
        assert_eq!(got.report.within_bound(), Some(true), "{ic}x{oc} g{gs}");
        assert!(got.report.max_abs > 0.0);
        assert_eq!(got.output.z.iter().filter(|&&z| z > 15).count(), 0);
        assert_eq!(got.output.q.iter().filter(|&&q| q > 15).count(), 0);
    }
}

#[test]
fn the_output_is_a_valid_layer_the_hf_packer_accepts() {
    let base = integer_base(128, 8, 128);
    let got = merge_lora_into_awq(&base, &adapter(8, 128, 2, 1.0, 9)).unwrap();
    let hf = ml_convertant_awq::pack_hf(&got.output).unwrap();
    assert_eq!(ml_convertant_awq::unpack_hf(&hf).unwrap(), got.output);
}

#[test]
fn geometry_mistakes_and_nonfinite_values_are_refused() {
    let base = integer_base(64, 8, 32);
    assert!(matches!(
        merge_lora_into_awq(&base, &adapter(8, 96, 2, 1.0, 1)),
        Err(RequantError::Lora(_))
    ));
    assert!(matches!(
        merge_lora_into_awq(&base, &adapter(4, 64, 2, 1.0, 1)),
        Err(RequantError::Lora(_))
    ));
    let mut nan = adapter(8, 64, 2, 1.0, 1);
    nan.scale = f32::NAN;
    assert!(matches!(
        merge_lora_into_awq(&base, &nan),
        Err(RequantError::Format(AwqRequantError::NonFinite { .. }))
    ));
}

#[test]
fn rounding_the_scale_up_covers_the_whole_range() {
    for x in [1.0e-6f32, 0.0123, 0.5, 1.0, 3.7, 100.3] {
        let s = f16_round_up(x).unwrap().to_f32();
        assert!(s >= x, "{x} -> {s}");
        assert!(s <= x * (1.0 + 2f32.powi(-10)) + 1.0e-7, "{x} -> {s}");
    }
    assert!(f16_round_up(1.0e9).is_none());
}
