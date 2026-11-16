//! Real-checkpoint checks for the AWQ, GPTQ and calibration crates.
//!
//! Corpus-gated: the models live in `C:/Models` (override with `MLCONV_MODELS_DIR`) and are absent on
//! hosted CI, so each test prints `SKIPPED` and returns rather than failing.
//!
//! What is REAL and what is SYNTHETIC, per test:
//! - `awq_real_checkpoint`: real AutoAWQ output (Qwen3-0.6B-AWQ, every projection of every layer).
//! - `gptq_container_on_real_awq_codes`: real 4-bit codes/zeros/scales in a SYNTHETIC GPTQ container (no
//!   real GPTQ checkpoint is staged); it checks pack/unpack on real value distributions only.
//! - `calibrate_on_real_weights`: real SmolLM2-360M weights, SYNTHETIC activations (no captured
//!   activations are staged; an imatrix holds sums of squares, not activations).

use half::{bf16, f16};
use ml_convertant_awq::{
    HfAwq, dequantize_kernel, dequantize_logical, pack_hf, repack_for_kernel, unpack_hf,
};
use ml_convertant_calibrate::{AwqSearch, CalibrationSet, rtn_awq, search_awq};
use ml_convertant_gptq::{
    HfGptq, LogicalGptq, pack_hf as gptq_pack, to_awq_logical, unpack_hf as gptq_unpack,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    std::env::var_os("MLCONV_MODELS_DIR").map_or_else(|| PathBuf::from("C:/Models"), PathBuf::from)
}

struct Safetensors {
    file: File,
    base: u64,
    index: HashMap<String, (String, Vec<usize>, u64, u64)>,
}

impl Safetensors {
    fn open(dir: &str) -> Option<Self> {
        let mut file = File::open(models_dir().join(dir).join("model.safetensors")).ok()?;
        let mut n = [0u8; 8];
        file.read_exact(&mut n).ok()?;
        let n = u64::from_le_bytes(n);
        let mut hdr = vec![0u8; n as usize];
        file.read_exact(&mut hdr).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&hdr).ok()?;
        let mut index = HashMap::new();
        for (k, t) in v.as_object()? {
            if k == "__metadata__" {
                continue;
            }
            let shape = t["shape"]
                .as_array()?
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let off = t["data_offsets"].as_array()?;
            index.insert(
                k.clone(),
                (
                    t["dtype"].as_str()?.to_string(),
                    shape,
                    off[0].as_u64()?,
                    off[1].as_u64()?,
                ),
            );
        }
        Some(Self {
            file,
            base: 8 + n,
            index,
        })
    }

    fn bytes(&mut self, name: &str, dtype: &str) -> (Vec<usize>, Vec<u8>) {
        let (dt, shape, a, b) = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .clone();
        assert_eq!(dt, dtype, "{name} dtype");
        let mut buf = vec![0u8; (b - a) as usize];
        self.file.seek(SeekFrom::Start(self.base + a)).unwrap();
        self.file.read_exact(&mut buf).unwrap();
        (shape, buf)
    }

    fn u32s(&mut self, name: &str) -> (Vec<usize>, Vec<u32>) {
        let (s, b) = self.bytes(name, "I32");
        (
            s,
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect(),
        )
    }

    fn f16s(&mut self, name: &str) -> Vec<f16> {
        let (_, b) = self.bytes(name, "F16");
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
            .collect()
    }

    fn bf16_f32(&mut self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (s, b) = self.bytes(name, "BF16");
        (
            s,
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
                .collect(),
        )
    }
}

const PROJ: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

fn awq_layer(st: &mut Safetensors, prefix: &str) -> HfAwq {
    let (qs, qweight) = st.u32s(&format!("{prefix}.qweight"));
    let (zs, qzeros) = st.u32s(&format!("{prefix}.qzeros"));
    let scales = st.f16s(&format!("{prefix}.scales"));
    let (ic, oc) = (qs[0], qs[1] * 8);
    assert_eq!(zs[1], qs[1]);
    HfAwq {
        ic,
        oc,
        group_size: ic / zs[0],
        qweight,
        qzeros,
        scales,
    }
}

#[test]
fn awq_real_checkpoint() {
    let Some(mut st) = Safetensors::open("Qwen3-0.6B-AWQ") else {
        println!(
            "SKIPPED: Qwen3-0.6B-AWQ not under {}",
            models_dir().display()
        );
        return;
    };
    let mut layers = 0usize;
    let mut elements = 0usize;
    let mut weight_sq = 0.0f64;
    let mut checked_sq_n = 0usize;
    for l in 0..28 {
        for p in PROJ {
            let hf = awq_layer(&mut st, &format!("model.layers.{l}.{p}"));
            assert_eq!(
                hf.group_size, 128,
                "layer {l} {p}: config.json says group_size 128"
            );
            let logical = unpack_hf(&hf).unwrap_or_else(|e| panic!("layer {l} {p}: {e}"));
            // Codes and zero points are 4-bit; every scale is a finite positive f16.
            assert!(
                logical.q.iter().chain(&logical.z).all(|&c| c < 16),
                "layer {l} {p}: code out of 0..16"
            );
            assert!(
                logical
                    .scales
                    .iter()
                    .all(|s| s.to_f32().is_finite() && s.to_f32() > 0.0),
                "layer {l} {p}: scale"
            );
            // Both nibble orders must round-trip to the stored words: pack(unpack(x)) == x.
            let back = pack_hf(&logical).unwrap();
            assert_eq!(
                back.qweight, hf.qweight,
                "layer {l} {p}: qweight round trip"
            );
            assert_eq!(back.qzeros, hf.qzeros, "layer {l} {p}: qzeros round trip");
            // Kernel repack moves values unchanged: dequantized outputs identical.
            let deq = dequantize_logical(&logical).unwrap();
            let k = repack_for_kernel(&logical).unwrap();
            let kd = dequantize_kernel(&k).unwrap();
            assert_eq!(deq.len(), kd.len());
            assert!(
                deq.iter().zip(&kd).all(|(a, b)| a.to_bits() == b.to_bits()),
                "layer {l} {p}: repack changed a value"
            );
            // Non-vacuity: a real layer uses (nearly) the whole 4-bit range and is not constant.
            let mut seen = [false; 16];
            logical.q.iter().for_each(|&c| seen[c as usize] = true);
            assert!(
                seen.iter().filter(|&&s| s).count() >= 14,
                "layer {l} {p}: codes degenerate"
            );
            // Min-max grouping: nearly every group x column spans codes 0 and 15 (AutoAWQ quantizes min-max).
            let groups = hf.ic / hf.group_size;
            let mut spanning = 0usize;
            for g in 0..groups {
                for o in 0..hf.oc {
                    let col: Vec<u8> = (0..hf.group_size)
                        .map(|r| logical.q[(g * hf.group_size + r) * hf.oc + o])
                        .collect();
                    spanning += usize::from(col.contains(&0) && col.contains(&15));
                }
            }
            let frac = spanning as f64 / (groups * hf.oc) as f64;
            assert!(
                frac > 0.9,
                "layer {l} {p}: only {frac:.3} of group-columns span 0..15"
            );
            weight_sq += deq.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            checked_sq_n += deq.len();
            elements += hf.ic * hf.oc;
            layers += 1;
        }
    }
    assert_eq!(layers, 196, "28 layers x 7 projections");
    let rms = (weight_sq / checked_sq_n as f64).sqrt();
    println!("awq_real_checkpoint: {layers} layers, {elements} weights; dequantized rms {rms:.5}");
    assert!(
        (1e-3..1.0).contains(&rms),
        "dequantized rms {rms} is not weight-like"
    );
}

#[test]
fn gptq_container_on_real_awq_codes() {
    let Some(mut st) = Safetensors::open("Qwen3-0.6B-AWQ") else {
        println!(
            "SKIPPED: Qwen3-0.6B-AWQ not under {}",
            models_dir().display()
        );
        return;
    };
    let mut n = 0;
    for (l, p) in [
        (0, "self_attn.q_proj"),
        (13, "mlp.down_proj"),
        (27, "mlp.up_proj"),
    ] {
        let a = unpack_hf(&awq_layer(&mut st, &format!("model.layers.{l}.{p}"))).unwrap();
        let g = LogicalGptq {
            ic: a.ic,
            oc: a.oc,
            group_size: a.group_size,
            q: a.q.clone(),
            z: a.z.clone(),
            scales: a.scales.clone(),
            g_idx: None,
        };
        let hf: HfGptq = gptq_pack(&g).unwrap();
        let again = gptq_unpack(&hf).unwrap();
        assert_eq!(again, g, "layer {l} {p}: GPTQ pack/unpack");
        assert_eq!(
            to_awq_logical(&again).unwrap(),
            a,
            "layer {l} {p}: GPTQ -> AWQ layout"
        );
        n += 1;
    }
    println!("gptq_container_on_real_awq_codes: {n} layers (SYNTHETIC GPTQ container, real codes)");
}

/// Deterministic standard normals (SYNTHETIC calibration activations).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// SYNTHETIC activations: `n` rows, per-channel scale log-normal(0, 1) with every 37th channel an x20 outlier.
fn synthetic_activations(ic: usize, n: usize, seed: u64) -> CalibrationSet {
    let mut rng = Lcg(seed);
    let scale: Vec<f64> = (0..ic)
        .map(|i| rng.normal().exp() * if i % 37 == 0 { 20.0 } else { 1.0 })
        .collect();
    let x = (0..n * ic)
        .map(|k| (rng.normal() * scale[k % ic]) as f32)
        .collect();
    CalibrationSet::new(ic, x).unwrap()
}

#[test]
fn calibrate_on_real_weights() {
    let Some(mut st) = Safetensors::open("SmolLM2-360M-Instruct") else {
        println!(
            "SKIPPED: SmolLM2-360M-Instruct not under {}",
            models_dir().display()
        );
        return;
    };
    let mut gains = Vec::new();
    for l in [0usize, 15, 31] {
        for p in ["self_attn.q_proj", "mlp.up_proj", "mlp.down_proj"] {
            let (shape, w) = st.bf16_f32(&format!("model.layers.{l}.{p}.weight"));
            let (oc, ic) = (shape[0], shape[1]);
            let w_rms =
                (w.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / w.len() as f64).sqrt();
            assert!(
                w_rms > 1e-4 && w.iter().all(|v| v.is_finite()),
                "layer {l} {p}: real weights look wrong"
            );
            let cal = synthetic_activations(ic, 48, 0x5EED ^ (l as u64) << 8 ^ ic as u64);
            let rtn = rtn_awq(&w, oc, ic, 64, Some(&cal)).unwrap();
            assert_eq!(
                rtn.report.within_bound(),
                Some(true),
                "layer {l} {p}: RTN exceeded its proven bound"
            );
            let rel = rtn.report.rmse / w_rms;
            assert!(rel < 0.25, "layer {l} {p}: RTN relative weight rmse {rel}");
            let s = search_awq(
                &w,
                oc,
                ic,
                &cal,
                &AwqSearch {
                    group_size: 64,
                    grid: 20,
                },
            )
            .unwrap();
            assert_eq!(
                s.report.within_bound(),
                Some(true),
                "layer {l} {p}: search exceeded its bound"
            );
            // The search's baseline is round to nearest on the same activations: it must agree with rtn_awq.
            let base = rtn.output_report.as_ref().unwrap();
            assert!(
                (s.baseline_output_report.rmse - base.rmse).abs() <= 1e-6 * base.rmse.max(1.0),
                "layer {l} {p}: baseline != rtn_awq"
            );
            assert!(
                s.output_report.rmse <= s.baseline_output_report.rmse,
                "layer {l} {p}: search worse than RTN"
            );
            assert!(s.channel_scales.iter().all(|v| v.is_finite() && *v > 0.0));
            let gain = 1.0 - s.output_report.rmse / s.baseline_output_report.rmse;
            println!(
                "calibrate l{l} {p} [{oc}x{ic}] weight-rel-rmse {rel:.4} alpha {:.2} output-rmse gain {:.1}% (SYNTHETIC activations)",
                s.alpha,
                gain * 100.0
            );
            gains.push(gain);
        }
    }
    assert_eq!(gains.len(), 9);
}

/// Pack-order check against the dense checkpoint the AWQ one was made from (Qwen/Qwen3-0.6B).
///
/// AutoAWQ folds a per-INPUT-channel scale into each layer, so row `i` of the dequantized `[IC, OC]`
/// matrix is the dense column `i` times a constant (plus 4-bit noise). The Pearson correlation of those two
/// vectors, over the output channels, is therefore high when nibble order and scale/zero alignment are
/// right and near zero when output columns are permuted. The median over input rows is asserted.
#[test]
fn awq_pack_order_against_dense() {
    let (Some(mut awq), Some(mut dense)) = (
        Safetensors::open("Qwen3-0.6B-AWQ"),
        Safetensors::open("Qwen3-0.6B-dense"),
    ) else {
        println!(
            "SKIPPED: Qwen3-0.6B-AWQ or Qwen3-0.6B-dense not under {}",
            models_dir().display()
        );
        return;
    };
    let mut worst = 1.0f64;
    for (l, p) in [
        (0, "self_attn.q_proj"),
        (3, "self_attn.v_proj"),
        (9, "mlp.gate_proj"),
        (14, "mlp.down_proj"),
        (27, "mlp.up_proj"),
    ] {
        let logical = unpack_hf(&awq_layer(&mut awq, &format!("model.layers.{l}.{p}"))).unwrap();
        let (ic, oc) = (logical.ic, logical.oc);
        let deq = dequantize_logical(&logical).unwrap();
        let (shape, w) = dense.bf16_f32(&format!("model.layers.{l}.{p}.weight"));
        assert_eq!(shape, vec![oc, ic], "layer {l} {p}: dense shape");
        let mut corr: Vec<f64> = (0..ic)
            .map(|i| {
                let (a, b): (Vec<f64>, Vec<f64>) = (0..oc)
                    .map(|o| (f64::from(deq[i * oc + o]), f64::from(w[o * ic + i])))
                    .unzip();
                let (ma, mb) = (
                    a.iter().sum::<f64>() / oc as f64,
                    b.iter().sum::<f64>() / oc as f64,
                );
                let cov: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
                let (va, vb): (f64, f64) = (
                    a.iter().map(|x| (x - ma).powi(2)).sum(),
                    b.iter().map(|y| (y - mb).powi(2)).sum(),
                );
                if va > 0.0 && vb > 0.0 {
                    cov / (va * vb).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        corr.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let median = corr[corr.len() / 2];
        println!("pack-order l{l} {p} [{ic}x{oc}]: median row correlation with dense {median:.4}");
        worst = worst.min(median);
    }
    assert!(
        worst > 0.9,
        "median row correlation {worst} (a permuted column order gives about 0)"
    );
}

/// GPTQ against the dense model it was quantized from (SmolLM2-360M-Instruct), REAL checkpoint: AutoRound's
/// export in AutoGPTQ v1 format (4-bit, group 64, asymmetric, `desc_act` false; pinned commit in the PR body).
///
/// GPTQ stores no folded scales, so the dequantized `[IC, OC]` matrix is the dense weight transposed up to
/// 4-bit noise. Asserted per layer: median row correlation with dense, and relative RMSE against dense. A
/// wrong nibble order scrambles rows (correlation collapses); a wrong zero-point offset (the stored `z - 1`)
/// leaves correlation high but shifts every value by a whole scale, which the RMSE bound catches.
#[test]
fn gptq_real_checkpoint_against_dense() {
    let (Some(mut gptq), Some(mut dense)) = (
        Safetensors::open("SmolLM2-360M-Instruct-GPTQ-gs64-asym"),
        Safetensors::open("SmolLM2-360M-Instruct"),
    ) else {
        println!(
            "SKIPPED: SmolLM2-360M-Instruct-GPTQ-gs64-asym or SmolLM2-360M-Instruct not under {}",
            models_dir().display()
        );
        return;
    };
    let (mut worst_corr, mut worst_rel) = (1.0f64, 0.0f64);
    for l in [0usize, 11, 22, 31] {
        for p in PROJ {
            let name = format!("model.layers.{l}.{p}");
            let (qs, qweight) = gptq.u32s(&format!("{name}.qweight"));
            let (_, qzeros) = gptq.u32s(&format!("{name}.qzeros"));
            let (_, gi) = gptq.u32s(&format!("{name}.g_idx"));
            let scales = gptq.f16s(&format!("{name}.scales"));
            let (ic, oc) = (qs[0] * 8, qs[1]);
            let hf = HfGptq {
                ic,
                oc,
                group_size: 64,
                qweight,
                qzeros,
                scales,
                g_idx: Some(gi.iter().map(|&g| g as i32).collect()),
            };
            let logical = gptq_unpack(&hf).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                gptq_pack(&logical).unwrap().qweight,
                hf.qweight,
                "{name}: qweight round trip"
            );
            to_awq_logical(&logical)
                .unwrap_or_else(|e| panic!("{name}: {e} (g_idx is not sequential)"));
            let deq = ml_convertant_gptq::dequantize_logical(&logical).unwrap();
            let (shape, w) = dense.bf16_f32(&format!("{name}.weight"));
            assert_eq!(shape, vec![oc, ic], "{name}: dense shape");
            let mut corr: Vec<f64> = (0..ic)
                .map(|i| {
                    let (a, b): (Vec<f64>, Vec<f64>) = (0..oc)
                        .map(|o| (f64::from(deq[i * oc + o]), f64::from(w[o * ic + i])))
                        .unzip();
                    let (ma, mb) = (
                        a.iter().sum::<f64>() / oc as f64,
                        b.iter().sum::<f64>() / oc as f64,
                    );
                    let cov: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
                    let (va, vb): (f64, f64) = (
                        a.iter().map(|x| (x - ma).powi(2)).sum(),
                        b.iter().map(|y| (y - mb).powi(2)).sum(),
                    );
                    if va > 0.0 && vb > 0.0 {
                        cov / (va * vb).sqrt()
                    } else {
                        0.0
                    }
                })
                .collect();
            corr.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let median = corr[corr.len() / 2];
            let sse: f64 = (0..ic)
                .flat_map(|i| (0..oc).map(move |o| (i, o)))
                .map(|(i, o)| (f64::from(deq[i * oc + o]) - f64::from(w[o * ic + i])).powi(2))
                .sum();
            let wss: f64 = w.iter().map(|&v| f64::from(v).powi(2)).sum();
            let rel = (sse / wss).sqrt();
            println!(
                "gptq l{l} {p} [{ic}x{oc}]: median row corr {median:.4}, rel rmse vs dense {rel:.4}"
            );
            worst_corr = worst_corr.min(median);
            worst_rel = worst_rel.max(rel);
        }
    }
    assert!(worst_corr > 0.9, "median row correlation {worst_corr}");
    assert!(worst_rel < 0.25, "relative rmse {worst_rel}");
}
