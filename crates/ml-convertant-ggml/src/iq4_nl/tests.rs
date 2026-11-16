use super::*;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
fn layout_and_table() {
    assert_eq!(IQ4_NL_BLOCK_BYTES, 18);
    assert_eq!(KVALUES_IQ4NL[0], -127);
    assert_eq!(KVALUES_IQ4NL[15], 113);
    assert!(KVALUES_IQ4NL.windows(2).all(|w| w[0] < w[1]));
    // The widest gap is 24, which the error bound relies on.
    let widest = KVALUES_IQ4NL.windows(2).map(|w| w[1] - w[0]).max();
    assert_eq!(widest, Some(24));
}

#[test]
fn hand_worked_block() {
    // Only x[0] = -127 and x[16] = 113 carry weight (the weights are x^2), so the least-squares scale is exactly 1
    // and they land on table entries 0 and 15; the zeros land on the nearest entry to 0, which is 1 (index 8).
    let mut v = [0.0f32; 32];
    v[0] = -127.0;
    v[16] = 113.0;
    let q = quantize_iq4_nl(&v).unwrap();
    assert_eq!(&q[..2], &f16::from_f32(1.0).to_bits().to_le_bytes());
    assert_eq!(q[2], 0xF0);
    assert!(q[3..].iter().all(|&b| b == 0x88));
    let d = dequantize_iq4_nl(&q).unwrap();
    assert_eq!((d[0], d[16], d[1]), (-127.0, 113.0, 1.0));
}

#[test]
fn best_index_picks_the_nearest_and_breaks_ties_upward() {
    assert_eq!(best_index(-1000.0), 0);
    assert_eq!(best_index(1000.0), 15);
    assert_eq!(best_index(0.0), 8); // between -10 and 1
    assert_eq!(best_index(-4.5), 8); // -4.5 is 5.5 from both -10 and 1: a tie, which goes to the upper entry
    for (i, &v) in KVALUES_IQ4NL.iter().enumerate() {
        assert_eq!(best_index(f32::from(v)), i);
    }
}

#[test]
fn all_zero_and_vanishing_blocks_have_scale_zero_and_nibbles_8() {
    let mut v = vec![0.0f32; 64];
    for (j, x) in v.iter_mut().take(32).enumerate() {
        *x = (j as f32 - 15.5) * 0.37;
    }
    let q = quantize_iq4_nl(&v).unwrap();
    assert_eq!(&q[18..20], &[0, 0]);
    assert!(q[20..].iter().all(|&b| b == 0x88), "0 * x maps to entry 8");
    let d = dequantize_iq4_nl(&q).unwrap();
    assert!(d[32..].iter().all(|&x| x == 0.0));
    // Alone, negative zero, and below ggml's 1e-15 threshold, the block is the same.
    for block in [[0.0f32; 32], [-0.0; 32], [1.0e-16; 32]] {
        let z = quantize_iq4_nl(&block).unwrap();
        assert_eq!(&z[..2], &[0, 0]);
        assert!(z[2..].iter().all(|&b| b == 0x88));
    }
}

#[test]
fn rejects_bad_input() {
    assert_eq!(
        quantize_iq4_nl(&[0.0; 31]),
        Err(Iq4Error::Block(BlockError::NotBlockAligned { len: 31 }))
    );
    let mut v = [0.0f32; 32];
    v[3] = f32::NAN;
    assert_eq!(
        quantize_iq4_nl(&v),
        Err(Iq4Error::Block(BlockError::NonFinite { index: 3 }))
    );
    let mut big = [1.0f32; 32];
    big[0] = 1.0e30;
    assert_eq!(
        quantize_iq4_nl(&big),
        Err(Iq4Error::NonFiniteScale { block: 0 })
    );
    assert_eq!(
        dequantize_iq4_nl(&[0; 17]),
        Err(Iq4Error::Block(BlockError::BadByteLength {
            len: 17,
            block_bytes: 18
        }))
    );
}

#[test]
fn bound_holds_on_random_blocks_including_outliers() {
    let mut s = 11u64;
    for round in 0..3000 {
        let scale = 10f32.powi((round % 9) - 4);
        let mut v: Vec<f32> = (0..64).map(|_| lcg(&mut s) * scale).collect();
        if round % 3 == 0 {
            v[5] *= 50.0; // an outlier far outside the table's span after the weighted fit
        }
        let c = Iq4Nl::default().convert(&v).unwrap();
        assert_eq!(c.report.within_bound(), Some(true), "round {round}");
    }
}

#[test]
fn importance_validation_and_effect() {
    let mut s = 5u64;
    let v: Vec<f32> = (0..64).map(|_| lcg(&mut s)).collect();
    let bad = |w: &[f32]| {
        matches!(
            quantize_iq4_nl_weighted(&v, Some(w)),
            Err(Iq4Error::BadImportance { .. })
        )
    };
    assert!(bad(&[]));
    assert!(bad(&[1.0; 31])); // not a multiple of the block
    assert!(bad(&[1.0; 96])); // does not divide the input into rows
    let mut neg = vec![1.0f32; 32];
    neg[3] = -1.0;
    assert!(bad(&neg));
    neg[3] = f32::NAN;
    assert!(bad(&neg));
    // A row of 32 repeated over two rows, and a full-length vector of the same values, encode identically.
    let w32: Vec<f32> = (0..32).map(|i| 1.0 + i as f32).collect();
    let w64: Vec<f32> = w32.iter().chain(&w32).copied().collect();
    assert_eq!(
        quantize_iq4_nl_weighted(&v, Some(&w32)).unwrap(),
        quantize_iq4_nl_weighted(&v, Some(&w64)).unwrap()
    );
    // Weighting is not a no-op, and the conversion still reports a bound it meets.
    assert_ne!(
        quantize_iq4_nl_weighted(&v, Some(&w32)).unwrap(),
        quantize_iq4_nl(&v).unwrap()
    );
    let c = Iq4Nl {
        importance: Some(&w32),
    }
    .convert(&v)
    .unwrap();
    assert_eq!(c.report.within_bound(), Some(true));
}

#[test]
fn zeros_round_trip_exactly() {
    let c = Iq4Nl::default().convert(&[0.0; 64]).unwrap();
    assert_eq!(c.report.max_abs, 0.0);
}
