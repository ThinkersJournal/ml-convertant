//! Differential against Fuel's independently written, end-to-end-validated repack
//! (`fuel-quantized/src/awq_repack.rs`): its output on seeded random HuggingFace-layout words is
//! recorded in `golden/fuel_repack.bin` (see `tools/golden/fuel_awq/README.md`); ours must be
//! identical word for word.

use half::f16;
use ml_convertant_awq::{HfAwq, HfToKernel, repack_for_kernel, unpack_hf};
use ml_convertant_core::Conversion;

const GOLDEN: &[u8] = include_bytes!("golden/fuel_repack.bin");

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let (h, t) = self.0.split_at(4);
        self.0 = t;
        u32::from_le_bytes(h.try_into().unwrap())
    }
    fn words(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.u32()).collect()
    }
    fn halves(&mut self, n: usize) -> Vec<f16> {
        (0..n)
            .map(|_| {
                let (h, t) = self.0.split_at(2);
                self.0 = t;
                f16::from_bits(u16::from_le_bytes(h.try_into().unwrap()))
            })
            .collect()
    }
}

#[test]
fn repack_is_identical_to_fuels_on_every_recorded_case() {
    let mut r = Reader(GOLDEN);
    let cases = r.u32();
    assert_eq!(cases, 5);
    for case in 0..cases {
        let (ic, oc, g, zeros_w) = (
            r.u32() as usize,
            r.u32() as usize,
            r.u32() as usize,
            r.u32() as usize,
        );
        let lens: Vec<usize> = (0..6).map(|_| r.u32() as usize).collect();
        let qweight = r.words(lens[0]);
        let qzeros = r.words(lens[1]);
        let want_weight = r.words(lens[3]);
        let want_zeros = r.words(lens[4]);
        let scales = r.halves(lens[2]);
        let want_scales = r.halves(lens[5]);

        let hf = HfAwq {
            ic,
            oc,
            group_size: g,
            qweight,
            qzeros,
            scales,
        };
        let got = repack_for_kernel(&unpack_hf(&hf).unwrap()).unwrap();
        assert_eq!(got.zeros_w, zeros_w, "case {case} zeros_w");
        assert_eq!(got.weight, want_weight, "case {case} weight");
        assert_eq!(got.zeros, want_zeros, "case {case} zeros");
        let bits = |v: &[f16]| v.iter().map(|h| h.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got.scales), bits(&want_scales), "case {case} scales");

        let c = HfToKernel.convert(&hf).unwrap();
        assert_eq!(c.output, got);
        assert_eq!(c.report.max_abs, 0.0);
    }
    assert!(r.0.is_empty(), "trailing bytes");
}
