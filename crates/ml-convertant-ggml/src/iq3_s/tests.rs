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

#[test]
fn layout_and_grid() {
    assert_eq!(IQ3_S_BLOCK_BYTES, 110);
    // 512 distinct points, each byte an odd number in 1..=15.
    let mut seen = std::collections::HashSet::new();
    for j in 0..512 {
        let p = point(j);
        assert!(p.iter().all(|v| (1..=15).contains(v) && v % 2 == 1));
        assert!(seen.insert(p));
    }
    let t = tables();
    assert_eq!(t.map.iter().filter(|m| **m >= 0).count(), 512);
    // Every off-grid quadruple has a non-empty neighbour list, ordered by distance, containing a nearest point.
    for (u, nb) in t.neighbours.iter().enumerate() {
        assert_eq!(nb.is_empty(), t.map[u] >= 0);
    }
    for (u, nb) in t
        .neighbours
        .iter()
        .enumerate()
        .filter(|(u, _)| t.map[*u] < 0)
    {
        let pos: [i32; 4] = std::array::from_fn(|k| 2 * ((u >> (3 * k)) & 7) as i32 + 1);
        let dist = |j: usize| -> i32 { (0..4).map(|k| (point(j)[k] - pos[k]).pow(2)).sum() };
        let best = (0..512).map(dist).min().unwrap();
        assert_eq!(dist(usize::from(nb[0])), best);
        assert!(
            nb.windows(2)
                .all(|w| (dist(usize::from(w[0])), w[0]) < (dist(usize::from(w[1])), w[1]))
        );
    }
}

#[test]
fn signs_follow_the_input() {
    let x = random(256, 11, 1.0);
    let d = dequantize_iq3_s(&quantize_iq3_s(&x).unwrap()).unwrap();
    for (a, b) in x.iter().zip(&d) {
        assert!(
            *a == 0.0 || a.is_sign_negative() == b.is_sign_negative(),
            "{a} {b}"
        );
    }
}

#[test]
fn all_zero_block_is_zero_bytes() {
    for block in [[0.0f32; 256], [-0.0; 256]] {
        assert_eq!(quantize_iq3_s(&block).unwrap(), vec![0u8; 110]);
    }
}

#[test]
fn round_trip_error_is_small_and_reported_without_a_bound() {
    for seed in 1..10u64 {
        let x = random(1024, seed, 10f32.powi((seed % 5) as i32 - 2));
        let r = Iq3S::default().convert(&x).unwrap().report;
        assert_eq!(r.bound, None);
        assert!(r.is_clean());
        // 3.4 bits per weight on uniform data: relative RMS error well under the signal.
        let rms = (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        assert!(
            r.rmse < 0.25 * rms,
            "seed {seed}: rmse {} vs rms {rms}",
            r.rmse
        );
    }
}

#[test]
fn importance_repeats_per_row_and_changes_the_result() {
    let x = random(1024, 3, 1.0);
    let mut w = vec![1.0f32; 512];
    for v in &mut w[..32] {
        *v = 0.0;
    }
    let weighted = quantize_iq3_s_weighted(&x, Some(&w)).unwrap();
    assert_ne!(weighted, quantize_iq3_s(&x).unwrap());
    let second = quantize_iq3_s_weighted(&x[512..], Some(&w)).unwrap();
    assert_eq!(&weighted[2 * IQ3_S_BLOCK_BYTES..], &second[..]);
}

#[test]
fn rejects_bad_input() {
    assert!(matches!(
        quantize_iq3_s(&[0.0; 255]),
        Err(Iq4Error::Block(BlockError::NotBlockAligned { len: 255 }))
    ));
    let mut x = vec![1.0f32; 256];
    x[9] = f32::INFINITY;
    assert!(matches!(
        quantize_iq3_s(&x),
        Err(Iq4Error::Block(BlockError::NonFinite { index: 9 }))
    ));
    let x = vec![1.0f32; 512];
    for w in [vec![], vec![1.0; 255], vec![1.0; 768], vec![-1.0; 256]] {
        assert!(matches!(
            quantize_iq3_s_weighted(&x, Some(&w)),
            Err(Iq4Error::BadImportance { .. })
        ));
    }
    assert!(matches!(
        dequantize_iq3_s(&[0u8; 109]),
        Err(Iq4Error::Block(BlockError::BadByteLength {
            len: 109,
            block_bytes: 110
        }))
    ));
}

#[test]
fn decode_reads_the_fields_where_ggml_does() {
    // Hand-built block: d = 1, grid index 0 = (1,1,1,1) everywhere except element group 0 of sub-block 3, which uses
    // grid index 256 + 5 (ninth bit via qh[3] bit 0) and has its signs set; sub-block 3's scale nibble is 5.
    let mut b = [0u8; 110];
    b[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
    b[2 + 3 * 8] = 5;
    b[66 + 3] = 1;
    b[74 + 3 * 4] = 0xFF;
    b[106 + 1] = 5 << 4; // sub-block 3 is the high nibble of scales[1]
    let d = dequantize_iq3_s(&b).unwrap();
    let g = point(256 + 5);
    for j in 0..4 {
        assert_eq!(d[3 * 32 + j], -(11.0 * g[j] as f32));
    }
    assert_eq!(d[3 * 32 + 4], -11.0); // second group of that byte: grid 0, scale 1 + 2 * 5, negative
    assert_eq!(d[0], 1.0);
}
