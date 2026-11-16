use super::*;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * scale).collect()
}

/// Scale code `ls` (0..=63) of sub-block `ib` read back from the bytes of block 0.
fn code(q: &[u8], ib: usize) -> usize {
    let h = u16::from_le_bytes([q[2], q[3]]);
    usize::from((q[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (usize::from((h >> (2 * ib)) & 3) << 4)
}

#[test]
fn layout() {
    assert_eq!(IQ4_XS_BLOCK_BYTES, 136);
    assert_eq!(QK_IQ4_XS, 256);
}

#[test]
fn scale_fields_are_packed_as_ggml_packs_them() {
    // Sub-block 0 is the loudest with a positive largest element (scale > 0 so l = -32, code 0); sub-block 1 has
    // the opposite sign and half the size (l = +16, code 48); sub-block 2 is silent (l = 0, code 32).
    let mut x = vec![0.0f32; 256];
    for j in 0..32 {
        x[j] = 100.0 * (1.0 + j as f32 / 31.0);
        x[32 + j] = -50.0 * (1.0 + j as f32 / 31.0);
    }
    let q = quantize_iq4_xs(&x).unwrap();
    assert_eq!(code(&q, 0), 0);
    assert_eq!(code(&q, 2), 32);
    let c1 = code(&q, 1);
    assert!(c1 > 32, "opposite sign gives a positive l: code {c1}");
    // Silent sub-blocks decode to exactly zero and carry nibble 8.
    assert!(q[8 + 64..8 + 96].iter().all(|&b| b == 0x88));
    let d = dequantize_iq4_xs(&q).unwrap();
    assert!(d[64..].iter().all(|&v| v == 0.0));
}

#[test]
fn all_zero_block_has_negative_zero_d_and_nibbles_8() {
    // ggml stores d = -0 / 32, which is f16 negative zero (0x8000); every scale code is 32 and every nibble 8.
    for block in [[0.0f32; 256], [-0.0; 256], [1.0e-16; 256]] {
        let q = quantize_iq4_xs(&block).unwrap();
        assert_eq!(&q[..2], &[0x00, 0x80]);
        assert!((0..8).all(|ib| code(&q, ib) == 32));
        assert!(q[8..].iter().all(|&b| b == 0x88));
        assert!(dequantize_iq4_xs(&q).unwrap().iter().all(|v| *v == 0.0));
    }
}

#[test]
fn round_trip_is_within_the_proven_bound() {
    for seed in 1..40u64 {
        let x = random(512, seed, 10f32.powi((seed % 7) as i32 - 3));
        let c = Iq4Xs::default().convert(&x).unwrap();
        let r = c.report;
        assert_eq!(r.within_bound(), Some(true), "seed {seed}");
        assert!(r.is_clean());
    }
}

#[test]
fn error_is_smaller_than_one_scale_per_block() {
    // Sanity on quality, not only on bytes: a 4-bit non-linear block keeps the relative RMS error small.
    let x = random(4096, 7, 1.0);
    let r = Iq4Xs::default().convert(&x).unwrap().report;
    assert!(r.rmse < 0.05, "rmse {}", r.rmse);
}

#[test]
fn importance_repeats_per_row_and_changes_the_result() {
    let x = random(1024, 3, 1.0);
    let mut w = vec![1.0f32; 512];
    for v in &mut w[..32] {
        *v = 0.0;
    }
    let weighted = quantize_iq4_xs_weighted(&x, Some(&w)).unwrap();
    let plain = quantize_iq4_xs(&x).unwrap();
    assert_ne!(weighted, plain);
    // The second row (elements 512..) uses the same vector from its start: encoding it alone gives its bytes.
    let second = quantize_iq4_xs_weighted(&x[512..], Some(&w)).unwrap();
    assert_eq!(&weighted[2 * IQ4_XS_BLOCK_BYTES..], &second[..]);
}

#[test]
fn rejects_bad_input() {
    assert!(matches!(
        quantize_iq4_xs(&[0.0; 255]),
        Err(Iq4Error::Block(BlockError::NotBlockAligned { len: 255 }))
    ));
    let mut x = vec![1.0f32; 256];
    x[9] = f32::NAN;
    assert!(matches!(
        quantize_iq4_xs(&x),
        Err(Iq4Error::Block(BlockError::NonFinite { index: 9 }))
    ));
    let x = vec![1.0f32; 512];
    for w in [
        vec![],
        vec![1.0; 255],
        vec![1.0; 768],
        vec![-1.0; 256],
        vec![f32::NAN; 256],
    ] {
        assert!(
            matches!(
                quantize_iq4_xs_weighted(&x, Some(&w)),
                Err(Iq4Error::BadImportance { .. })
            ),
            "len {}",
            w.len()
        );
    }
    assert!(matches!(
        dequantize_iq4_xs(&[0u8; 135]),
        Err(Iq4Error::Block(BlockError::BadByteLength {
            len: 135,
            block_bytes: 136
        }))
    ));
}

#[test]
fn overflowing_scale_is_an_error_and_f16_overflow_gives_an_infinite_bound() {
    // Squaring 1e20 overflows f32.
    assert!(matches!(
        quantize_iq4_xs(&[1.0e20; 256]),
        Err(Iq4Error::NonFiniteScale { block: 0 })
    ));
    // d = max_scale / 32 beyond f16's range: stored as infinity, so the bound is infinite and the decode is not finite.
    let mut x = vec![1.0e9f32; 256];
    x[0] = 1.0e10;
    let c = Iq4Xs::default().convert(&x).unwrap();
    assert_eq!(c.report.bound, Some(f64::INFINITY));
    assert!(c.report.nonfinite_out > 0);
}

#[test]
fn half_way_sub_block_scales_round_to_even() {
    // Sub-block 0 has scale 64, so d = -2 and its l is -32; sub-block 1 has scale 33, so l = -16.5, which
    // rounds to the even -16 (code 16; round-half-away would give -17); sub-block 2 has scale 35, l = -17.5,
    // which rounds to the even -18 (code 14; round-half-up would give -17).
    let mut x = vec![0.0f32; 256];
    for (ib, a) in [(0usize, 64.0f32), (1, 33.0), (2, 35.0)] {
        x[ib * 32] = -127.0 * a;
        x[ib * 32 + 16] = 113.0 * a;
    }
    let q = quantize_iq4_xs(&x).unwrap();
    let codes: Vec<usize> = (0..3).map(|ib| code(&q, ib)).collect();
    assert_eq!(codes[0], 0);
    assert_eq!(codes[1], 16, "-16.5 rounds to -16");
    assert_eq!(codes[2], 14, "-17.5 rounds to -18");
}
