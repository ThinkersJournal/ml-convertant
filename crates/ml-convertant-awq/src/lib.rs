//! AWQ int4 (GEMM version) pack, unpack and kernel repack, with measured error.
//!
//! # Layouts
//!
//! A linear layer has `IC` input and `OC` output features, quantized in groups of `group_size`
//! along `IC`. Logical data are 4-bit codes `q[IC, OC]`, zero points `z[IC/g, OC]` and f16 scales
//! `s[IC/g, OC]`; an element dequantizes to `(q - z) * s`, rounded to f16 as AutoAWQ's
//! `dequantize_gemm` computes it in half precision.
//!
//! **HuggingFace / AutoAWQ** ([`HfAwq`]): `qweight [IC, OC/8]` and `qzeros [IC/g, OC/8]` as `u32`,
//! `scales [IC/g, OC]` as f16. Eight consecutive output columns share a word; nibble `i` of word `col`
//! holds logical column `col * 8 + HF_AWQ_PACK_ORDER[i]` (AutoAWQ's `order_map`).
//!
//! **Kernel** ([`KernelAwq`]): everything OC-major with `groups = IC / g` and `mult = 128 / g`:
//! `weight [OC, IC/8]` (nibble `ic % 8` of word `ic / 8`), `zeros [OC, zeros_w]` (nibble `g % 8` of
//! word `g / 8`), `scales [OC, 8 * zeros_w]` (entry `g`), with
//! `zeros_w = ceil(ceil(groups / 8) / mult) * mult`; the tail of each `zeros`/`scales` row past
//! `groups` is zero padding. This is the layout documented on baracuda's `Int4AwqGemmArgs` and
//! consumed by Fuel's W4A16 AWQ kernel; `group_size` must be 64 or 128 here.
//!
//! # Error
//!
//! Every conversion here moves codes, zeros and scales without changing them, so the error is exactly
//! zero. [`HfToKernel`] proves it per layer rather than assuming it: it dequantizes the input and the
//! output independently and reports their difference (always `max_abs == 0`, bound `0`).

use half::f16;
use ml_convertant_core::{Conversion, Converted, ErrorReport, LengthMismatch};
use std::fmt;

/// AutoAWQ's `order_map` (`WQLinear_GEMM.from_linear`): nibble position `i` of a packed word holds
/// logical column `HF_AWQ_PACK_ORDER[i]` within its group of eight.
pub const HF_AWQ_PACK_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Why an AWQ conversion failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwqError {
    /// `group_size` was zero, or not 64/128 where the kernel layout needs it.
    BadGroupSize(usize),
    /// A dimension is not a multiple of what the packing requires.
    BadDim {
        /// Which dimension.
        what: &'static str,
        /// Its value.
        value: usize,
        /// What it must be a multiple of.
        multiple_of: usize,
    },
    /// A buffer has the wrong length.
    BadLen {
        /// Which buffer.
        what: &'static str,
        /// Its length.
        got: usize,
        /// The expected length.
        want: usize,
    },
    /// A code or zero point does not fit in 4 bits.
    CodeOutOfRange {
        /// Which buffer.
        what: &'static str,
        /// Index of the offending element.
        index: usize,
    },
    /// Measuring failed (cannot happen for equal-length slices).
    Measure(LengthMismatch),
}

impl fmt::Display for AwqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadGroupSize(g) => write!(f, "unsupported AWQ group size {g}"),
            Self::BadDim {
                what,
                value,
                multiple_of,
            } => write!(
                f,
                "AWQ {what} = {value} must be a multiple of {multiple_of}"
            ),
            Self::BadLen { what, got, want } => {
                write!(f, "AWQ {what} has {got} elements, expected {want}")
            }
            Self::CodeOutOfRange { what, index } => {
                write!(f, "AWQ {what}[{index}] does not fit in 4 bits")
            }
            Self::Measure(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for AwqError {}

/// A layer as stored in a HuggingFace `*-AWQ` (GEMM) checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct HfAwq {
    /// Input features.
    pub ic: usize,
    /// Output features.
    pub oc: usize,
    /// Quantization group size along `IC`.
    pub group_size: usize,
    /// `[IC, OC/8]` packed weights.
    pub qweight: Vec<u32>,
    /// `[IC/group_size, OC/8]` packed zero points.
    pub qzeros: Vec<u32>,
    /// `[IC/group_size, OC]` scales.
    pub scales: Vec<f16>,
}

/// A layer as logical 4-bit codes, one per `u8`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalAwq {
    /// Input features.
    pub ic: usize,
    /// Output features.
    pub oc: usize,
    /// Quantization group size along `IC`.
    pub group_size: usize,
    /// `[IC, OC]` codes, each in `0..16`.
    pub q: Vec<u8>,
    /// `[IC/group_size, OC]` zero points, each in `0..16`.
    pub z: Vec<u8>,
    /// `[IC/group_size, OC]` scales.
    pub scales: Vec<f16>,
}

/// A layer in the OC-major kernel layout (see the crate docs).
#[derive(Debug, Clone, PartialEq)]
pub struct KernelAwq {
    /// Input features.
    pub ic: usize,
    /// Output features.
    pub oc: usize,
    /// Quantization group size along `IC` (64 or 128).
    pub group_size: usize,
    /// `[OC, IC/8]` packed weights.
    pub weight: Vec<u32>,
    /// `[OC, zeros_w]` packed zero points.
    pub zeros: Vec<u32>,
    /// `[OC, 8 * zeros_w]` scales.
    pub scales: Vec<f16>,
    /// Words per `zeros` row.
    pub zeros_w: usize,
}

fn check_len(what: &'static str, got: usize, want: usize) -> Result<(), AwqError> {
    if got == want {
        Ok(())
    } else {
        Err(AwqError::BadLen { what, got, want })
    }
}

fn check_multiple(what: &'static str, value: usize, multiple_of: usize) -> Result<(), AwqError> {
    if multiple_of != 0 && value.is_multiple_of(multiple_of) {
        Ok(())
    } else {
        Err(AwqError::BadDim {
            what,
            value,
            multiple_of,
        })
    }
}

/// Validates dimensions; returns the group count.
fn geometry(ic: usize, oc: usize, group_size: usize) -> Result<usize, AwqError> {
    if group_size == 0 {
        return Err(AwqError::BadGroupSize(0));
    }
    check_multiple("IC", ic, group_size)?;
    check_multiple("IC", ic, 8)?;
    check_multiple("OC", oc, 8)?;
    Ok(ic / group_size)
}

/// Words per kernel `zeros` row: `ceil(ceil(groups / 8) / mult) * mult`, `mult = 128 / group_size`.
pub fn kernel_zeros_w(ic: usize, group_size: usize) -> Result<usize, AwqError> {
    if group_size != 64 && group_size != 128 {
        return Err(AwqError::BadGroupSize(group_size));
    }
    check_multiple("IC", ic, group_size)?;
    let mult = 128 / group_size;
    Ok((ic / group_size).div_ceil(8).div_ceil(mult) * mult)
}

fn unpack_words(words: &[u32], rows: usize, oc: usize) -> Vec<u8> {
    let wpr = oc / 8;
    let mut out = vec![0u8; rows * oc];
    for r in 0..rows {
        for col in 0..wpr {
            let w = words[r * wpr + col];
            for (i, &logical) in HF_AWQ_PACK_ORDER.iter().enumerate() {
                out[r * oc + col * 8 + logical] = ((w >> (4 * i)) & 0xF) as u8;
            }
        }
    }
    out
}

fn pack_words(codes: &[u8], rows: usize, oc: usize) -> Vec<u32> {
    let wpr = oc / 8;
    let mut out = vec![0u32; rows * wpr];
    for r in 0..rows {
        for col in 0..wpr {
            for (i, &logical) in HF_AWQ_PACK_ORDER.iter().enumerate() {
                out[r * wpr + col] |= u32::from(codes[r * oc + col * 8 + logical]) << (4 * i);
            }
        }
    }
    out
}

/// Unpacks a HuggingFace layer to logical codes.
pub fn unpack_hf(hf: &HfAwq) -> Result<LogicalAwq, AwqError> {
    let groups = geometry(hf.ic, hf.oc, hf.group_size)?;
    check_len("qweight", hf.qweight.len(), hf.ic * (hf.oc / 8))?;
    check_len("qzeros", hf.qzeros.len(), groups * (hf.oc / 8))?;
    check_len("scales", hf.scales.len(), groups * hf.oc)?;
    Ok(LogicalAwq {
        ic: hf.ic,
        oc: hf.oc,
        group_size: hf.group_size,
        q: unpack_words(&hf.qweight, hf.ic, hf.oc),
        z: unpack_words(&hf.qzeros, groups, hf.oc),
        scales: hf.scales.clone(),
    })
}

/// Packs logical codes the way AutoAWQ's `WQLinear_GEMM.from_linear` does.
pub fn pack_hf(l: &LogicalAwq) -> Result<HfAwq, AwqError> {
    let groups = geometry(l.ic, l.oc, l.group_size)?;
    check_len("q", l.q.len(), l.ic * l.oc)?;
    check_len("z", l.z.len(), groups * l.oc)?;
    check_len("scales", l.scales.len(), groups * l.oc)?;
    if let Some(index) = l.q.iter().position(|&v| v > 15) {
        return Err(AwqError::CodeOutOfRange { what: "q", index });
    }
    if let Some(index) = l.z.iter().position(|&v| v > 15) {
        return Err(AwqError::CodeOutOfRange { what: "z", index });
    }
    Ok(HfAwq {
        ic: l.ic,
        oc: l.oc,
        group_size: l.group_size,
        qweight: pack_words(&l.q, l.ic, l.oc),
        qzeros: pack_words(&l.z, groups, l.oc),
        scales: l.scales.clone(),
    })
}

/// Repacks logical codes to the OC-major kernel layout (`group_size` 64 or 128).
pub fn repack_for_kernel(l: &LogicalAwq) -> Result<KernelAwq, AwqError> {
    let zeros_w = kernel_zeros_w(l.ic, l.group_size)?;
    let groups = geometry(l.ic, l.oc, l.group_size)?;
    check_len("q", l.q.len(), l.ic * l.oc)?;
    check_len("z", l.z.len(), groups * l.oc)?;
    check_len("scales", l.scales.len(), groups * l.oc)?;
    if let Some(index) = l.q.iter().position(|&v| v > 15) {
        return Err(AwqError::CodeOutOfRange { what: "q", index });
    }
    if let Some(index) = l.z.iter().position(|&v| v > 15) {
        return Err(AwqError::CodeOutOfRange { what: "z", index });
    }
    let (ic, oc) = (l.ic, l.oc);
    let mut weight = vec![0u32; oc * (ic / 8)];
    let mut zeros = vec![0u32; oc * zeros_w];
    let mut scales = vec![f16::ZERO; oc * zeros_w * 8];
    for o in 0..oc {
        for r in 0..ic {
            weight[o * (ic / 8) + r / 8] |= u32::from(l.q[r * oc + o]) << (4 * (r % 8));
        }
        for g in 0..groups {
            zeros[o * zeros_w + g / 8] |= u32::from(l.z[g * oc + o]) << (4 * (g % 8));
            scales[o * zeros_w * 8 + g] = l.scales[g * oc + o];
        }
    }
    Ok(KernelAwq {
        ic,
        oc,
        group_size: l.group_size,
        weight,
        zeros,
        scales,
        zeros_w,
    })
}

/// Dequantizes logical codes to `[IC, OC]` values: `(q - z) * s`, rounded to f16.
pub fn dequantize_logical(l: &LogicalAwq) -> Result<Vec<f32>, AwqError> {
    let groups = geometry(l.ic, l.oc, l.group_size)?;
    check_len("q", l.q.len(), l.ic * l.oc)?;
    check_len("z", l.z.len(), groups * l.oc)?;
    check_len("scales", l.scales.len(), groups * l.oc)?;
    let mut out = Vec::with_capacity(l.ic * l.oc);
    for r in 0..l.ic {
        let g = r / l.group_size;
        for o in 0..l.oc {
            let d = f32::from(l.q[r * l.oc + o]) - f32::from(l.z[g * l.oc + o]);
            out.push(f16::from_f32(d * l.scales[g * l.oc + o].to_f32()).to_f32());
        }
    }
    Ok(out)
}

/// Dequantizes kernel-layout buffers to `[IC, OC]` values, reading only the kernel layout.
pub fn dequantize_kernel(k: &KernelAwq) -> Result<Vec<f32>, AwqError> {
    let groups = geometry(k.ic, k.oc, k.group_size)?;
    check_len("weight", k.weight.len(), k.oc * (k.ic / 8))?;
    check_len("zeros", k.zeros.len(), k.oc * k.zeros_w)?;
    check_len("scales", k.scales.len(), k.oc * k.zeros_w * 8)?;
    let mut out = vec![0.0f32; k.ic * k.oc];
    for o in 0..k.oc {
        for r in 0..k.ic {
            let g = r / k.group_size;
            let q = (k.weight[o * (k.ic / 8) + r / 8] >> (4 * (r % 8))) & 0xF;
            let z = (k.zeros[o * k.zeros_w + g / 8] >> (4 * (g % 8))) & 0xF;
            let s = k.scales[o * k.zeros_w * 8 + g].to_f32();
            debug_assert!(g < groups);
            out[r * k.oc + o] = f16::from_f32((q as f32 - z as f32) * s).to_f32();
        }
    }
    Ok(out)
}

/// HuggingFace `*-AWQ` layer -> kernel layout, reporting the (zero) error between the dequantized
/// weights before and after.
#[derive(Debug, Clone, Copy, Default)]
pub struct HfToKernel;

impl Conversion for HfToKernel {
    type Input = HfAwq;
    type Output = KernelAwq;
    type Error = AwqError;

    fn convert(&self, input: &HfAwq) -> Result<Converted<KernelAwq>, AwqError> {
        let logical = unpack_hf(input)?;
        let output = repack_for_kernel(&logical)?;
        let before = dequantize_logical(&logical)?;
        let after = dequantize_kernel(&output)?;
        let report = ErrorReport::measure(&before, &after)
            .map_err(AwqError::Measure)?
            .with_bound(0.0);
        Ok(Converted { output, report })
    }
}

#[cfg(test)]
mod tests {
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

    pub(crate) fn logical(ic: usize, oc: usize, group: usize, seed: u64) -> LogicalAwq {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
        let groups = ic / group;
        LogicalAwq {
            ic,
            oc,
            group_size: group,
            q: (0..ic * oc).map(|_| (r.next() & 0xF) as u8).collect(),
            z: (0..groups * oc).map(|_| (r.next() & 0xF) as u8).collect(),
            scales: (0..groups * oc)
                .map(|i| f16::from_f32(0.01 + i as f32 * 0.001))
                .collect(),
        }
    }

    /// AutoAWQ's own formulation (`packing_utils.unpack_awq` + `reverse_awq_order`): read nibbles in
    /// sequence, then gather with `AWQ_REVERSE_ORDER`. Independent of `HF_AWQ_PACK_ORDER`.
    fn autoawq_unpack(words: &[u32], rows: usize, oc: usize) -> Vec<u8> {
        const REVERSE: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
        let wpr = oc / 8;
        let mut out = vec![0u8; rows * oc];
        for r in 0..rows {
            let seq: Vec<u8> = (0..oc)
                .map(|c| ((words[r * wpr + c / 8] >> (4 * (c % 8))) & 0xF) as u8)
                .collect();
            for c in 0..oc {
                out[r * oc + c] = seq[(c / 8) * 8 + REVERSE[c % 8]];
            }
        }
        out
    }

    #[test]
    fn hf_unpack_agrees_with_autoawqs_reverse_order_formulation() {
        let l = logical(256, 64, 128, 1);
        let hf = pack_hf(&l).unwrap();
        assert_eq!(autoawq_unpack(&hf.qweight, 256, 64), l.q);
        assert_eq!(autoawq_unpack(&hf.qzeros, 2, 64), l.z);
    }

    #[test]
    fn pack_unpack_round_trip() {
        for (ic, oc, g) in [(128, 8, 128), (256, 64, 64), (64, 24, 32)] {
            let l = logical(ic, oc, g, ic as u64);
            let back = unpack_hf(&pack_hf(&l).unwrap()).unwrap();
            assert_eq!(back, l);
        }
    }

    #[test]
    fn known_word() {
        // One row, eight columns holding 0..8: nibble i carries logical column ORDER[i].
        let mut l = logical(128, 8, 128, 2);
        l.q.iter_mut().for_each(|v| *v = 0);
        for c in 0..8 {
            l.q[c] = c as u8;
        }
        let hf = pack_hf(&l).unwrap();
        // nibbles low->high: columns 0,2,4,6,1,3,5,7
        assert_eq!(hf.qweight[0], 0x7531_6420);
    }

    #[test]
    fn kernel_zeros_w_matches_the_documented_formula() {
        assert_eq!(kernel_zeros_w(128, 128).unwrap(), 1);
        assert_eq!(kernel_zeros_w(1024, 128).unwrap(), 1);
        assert_eq!(kernel_zeros_w(1024, 64).unwrap(), 2);
        assert_eq!(kernel_zeros_w(128 * 9, 128).unwrap(), 2);
        assert_eq!(kernel_zeros_w(64 * 17, 64).unwrap(), 4);
        assert!(kernel_zeros_w(128, 32).is_err());
    }

    #[test]
    fn kernel_padding_is_zero_and_dequant_matches() {
        for (ic, oc, g) in [(256, 64, 128), (1024, 64, 64), (128 * 9, 16, 128)] {
            let l = logical(ic, oc, g, (ic + g) as u64);
            let k = repack_for_kernel(&l).unwrap();
            let groups = ic / g;
            for o in 0..oc {
                for j in groups..k.zeros_w * 8 {
                    assert_eq!(k.scales[o * k.zeros_w * 8 + j], f16::ZERO);
                    assert_eq!((k.zeros[o * k.zeros_w + j / 8] >> (4 * (j % 8))) & 0xF, 0);
                }
            }
            assert_eq!(
                dequantize_kernel(&k).unwrap(),
                dequantize_logical(&l).unwrap()
            );
        }
    }

    #[test]
    fn conversion_reports_exactly_zero_error() {
        let hf = pack_hf(&logical(256, 64, 128, 7)).unwrap();
        let c = HfToKernel.convert(&hf).unwrap();
        assert_eq!(c.report.max_abs, 0.0);
        assert_eq!(c.report.n_measured, 256 * 64);
        assert_eq!(c.report.within_bound(), Some(true));
        assert!(c.report.is_clean());
    }

    #[test]
    fn rejects_bad_input() {
        let l = logical(128, 8, 128, 3);
        let mut bad = l.clone();
        bad.q[0] = 16;
        assert_eq!(
            pack_hf(&bad),
            Err(AwqError::CodeOutOfRange {
                what: "q",
                index: 0
            })
        );
        let mut hf = pack_hf(&l).unwrap();
        hf.qweight.pop();
        assert!(matches!(unpack_hf(&hf), Err(AwqError::BadLen { .. })));
        let mut odd = l.clone();
        odd.oc = 12;
        assert!(matches!(pack_hf(&odd), Err(AwqError::BadDim { .. })));
        let mut g32 = logical(128, 8, 32, 4);
        assert!(repack_for_kernel(&g32).is_err());
        g32.group_size = 0;
        assert_eq!(pack_hf(&g32), Err(AwqError::BadGroupSize(0)));
    }
}
