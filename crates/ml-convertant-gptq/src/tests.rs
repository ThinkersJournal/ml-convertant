use super::*;

/// xorshift; no dependency needed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn logical(ic: usize, oc: usize, group: usize, seed: u64) -> LogicalGptq {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
    let groups = ic / group;
    LogicalGptq {
        ic,
        oc,
        group_size: group,
        q: (0..ic * oc).map(|_| (r.next() & 0xF) as u8).collect(),
        z: (0..groups * oc).map(|_| (r.next() & 0xF) as u8).collect(),
        scales: (0..groups * oc)
            .map(|i| f16::from_f32(0.01 + i as f32 * 0.001))
            .collect(),
        g_idx: None,
    }
}

/// AutoGPTQ's `pack`, transcribed independently: `intweight` is `[IC, OC]`, 8 values per word down the rows;
/// `zeros` is `[G, OC]`, decremented by one, 8 values per word along the columns.
fn autogptq_pack(l: &LogicalGptq) -> (Vec<u32>, Vec<u32>) {
    let mut qweight = vec![0u32; (l.ic / 8) * l.oc];
    let (mut i, mut row) = (0, 0);
    while row < l.ic / 8 {
        for j in i..i + 8 {
            for o in 0..l.oc {
                qweight[row * l.oc + o] |= u32::from(l.q[j * l.oc + o]) << (4 * (j - i));
            }
        }
        i += 8;
        row += 1;
    }
    let groups = l.ic / l.group_size;
    let mut qzeros = vec![0u32; groups * (l.oc / 8)];
    for g in 0..groups {
        let (mut i, mut col) = (0, 0);
        while col < l.oc / 8 {
            for j in i..i + 8 {
                let zeros_minus_one = u32::from(l.z[g * l.oc + j]).wrapping_sub(1) & 0xF;
                qzeros[g * (l.oc / 8) + col] |= zeros_minus_one << (4 * (j - i));
            }
            i += 8;
            col += 1;
        }
    }
    (qweight, qzeros)
}

#[test]
fn packing_matches_autogptqs_documented_pack() {
    for (ic, oc, gs, seed) in [(16, 8, 8, 1), (64, 16, 32, 2), (128, 8, 128, 3)] {
        let l = logical(ic, oc, gs, seed);
        let (qweight, qzeros) = autogptq_pack(&l);
        let hf = pack_hf(&l).unwrap();
        assert_eq!(hf.qweight, qweight, "{ic}x{oc} g{gs}");
        assert_eq!(hf.qzeros, qzeros, "{ic}x{oc} g{gs}");
    }
}

#[test]
fn unpack_inverts_pack_for_every_code_including_zero_point_zero() {
    for seed in 0..6 {
        let l = logical(64, 16, 32, seed);
        let back = unpack_hf(&pack_hf(&l).unwrap()).unwrap();
        assert_eq!(back, l);
    }
    // Effective zero 0 is stored as 15 (AutoGPTQ's mask after +1), and reads back as 0.
    let mut l = logical(16, 8, 16, 9);
    l.z.fill(0);
    let hf = pack_hf(&l).unwrap();
    assert_eq!(hf.qzeros, vec![0xFFFF_FFFFu32]);
    assert_eq!(unpack_hf(&hf).unwrap().z, l.z);
}

#[test]
fn a_hand_worked_word() {
    // 8 input features, 8 output features, one group. Column 0 of qweight = nibbles 0..8 (feature r -> r),
    // column 1 = all 15, others 0. Stored zeros nibble k = k, so effective zero k + 1 (7 -> 8).
    let mut qweight = vec![0u32; 8];
    qweight[0] = 0x7654_3210;
    qweight[1] = 0xFFFF_FFFF;
    let hf = HfGptq {
        ic: 8,
        oc: 8,
        group_size: 8,
        qweight,
        qzeros: vec![0x7654_3210],
        scales: vec![f16::from_f32(0.5); 8],
        g_idx: None,
    };
    let l = unpack_hf(&hf).unwrap();
    for r in 0..8 {
        assert_eq!(l.q[r * 8], r as u8);
        assert_eq!(l.q[r * 8 + 1], 15);
        assert_eq!(l.q[r * 8 + 2], 0);
    }
    assert_eq!(l.z, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let d = dequantize_logical(&l).unwrap();
    // feature 3, column 0: (3 - 1) * 0.5; feature 0, column 7: (0 - 8) * 0.5.
    assert_eq!(d[3 * 8], 1.0);
    assert_eq!(d[7], -4.0);
}

#[test]
fn dequantize_matches_autogptqs_scales_times_q_minus_zeros() {
    let l = logical(64, 16, 32, 7);
    let d = dequantize_logical(&l).unwrap();
    for r in 0..64 {
        for o in 0..16 {
            let g = r / 32;
            let stored = (i32::from(l.z[g * 16 + o]) + 15) & 15; // what pack writes
            let zeros = (stored + 1) & 15; // AutoGPTQ forward: zeros + 1, then & (2**bits - 1)
            let want = f16::from_f32(
                (i32::from(l.q[r * 16 + o]) - zeros) as f32 * l.scales[g * 16 + o].to_f32(),
            )
            .to_f32();
            assert_eq!(d[r * 16 + o], want);
        }
    }
}

#[test]
fn activation_order_dequantizes_by_g_idx_and_is_refused_by_the_awq_conversion() {
    let mut l = logical(16, 8, 8, 4);
    // Reverse the rows' groups: first 8 rows use group 1, last 8 use group 0.
    l.g_idx = Some((0..16).map(|r| if r < 8 { 1 } else { 0 }).collect());
    let d = dequantize_logical(&l).unwrap();
    let want =
        f16::from_f32((f32::from(l.q[0]) - f32::from(l.z[8])) * l.scales[8].to_f32()).to_f32();
    assert_eq!(d[0], want);
    assert_eq!(to_awq_logical(&l), Err(GptqError::ActivationOrder));
    // A sequential g_idx is accepted and changes nothing.
    l.g_idx = Some((0..16).map(|r| r / 8).collect());
    assert!(to_awq_logical(&l).is_ok());
    // An out-of-range group is an error, not an out-of-bounds read.
    l.g_idx = Some(vec![5; 16]);
    assert!(matches!(
        dequantize_logical(&l),
        Err(GptqError::BadGroupIndex { .. })
    ));
}

#[test]
fn gptq_to_awq_is_lossless_and_reports_exactly_zero() {
    let l = logical(128, 16, 64, 5);
    let hf = pack_hf(&l).unwrap();
    let got = GptqToAwq.convert(&hf).unwrap();
    assert_eq!(got.report.max_abs, 0.0);
    assert_eq!(got.report.within_bound(), Some(true));
    assert_eq!(got.output.q, l.q);
    assert_eq!(got.output.z, l.z);
    // The AWQ side dequantizes it identically, through its own HF packing as well.
    let awq_hf = ml_convertant_awq::pack_hf(&got.output).unwrap();
    let back = ml_convertant_awq::unpack_hf(&awq_hf).unwrap();
    assert_eq!(
        ml_convertant_awq::dequantize_logical(&back).unwrap(),
        dequantize_logical(&l).unwrap()
    );
}

#[test]
fn bad_geometry_and_codes_are_refused() {
    let mut l = logical(64, 16, 32, 1);
    l.q[3] = 16;
    assert!(matches!(
        pack_hf(&l),
        Err(GptqError::CodeOutOfRange {
            what: "q",
            index: 3
        })
    ));
    let mut hf = pack_hf(&logical(64, 16, 32, 1)).unwrap();
    hf.qweight.pop();
    assert!(matches!(unpack_hf(&hf), Err(GptqError::BadLen { .. })));
    assert!(matches!(
        unpack_hf(&HfGptq {
            ic: 64,
            oc: 12,
            group_size: 32,
            qweight: vec![],
            qzeros: vec![],
            scales: vec![],
            g_idx: None
        }),
        Err(GptqError::BadDim { what: "OC", .. })
    ));
}
