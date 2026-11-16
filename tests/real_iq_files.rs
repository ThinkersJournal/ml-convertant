//! Real-file checks for IQ4_NL, IQ4_XS, IQ3_S and IQ3_XXS: our decode against llama.cpp's own, and against Fuel's.
//!
//! Corpus-gated: the GGUF files live in `C:/Models` (override with `MLCONV_MODELS_DIR`) and are absent on
//! hosted CI, so the test prints `SKIPPED` and returns rather than failing.
//!
//! REAL: every IQ4_NL / IQ4_XS / IQ3_S / IQ3_XXS tensor (630 of them) of three published SmolLM2-135M-Instruct
//! GGUFs: the IQ3_XS file (IQ3_S + IQ4_NL), the IQ4_XS file (IQ4_XS + IQ4_NL) and an imatrix IQ3_XXS file
//! (IQ3_XXS + IQ4_NL; Apache-2.0, 86958272 bytes). Files are pinned by SHA-256 in `tests/golden/real_iq_dequant.tsv`.
//! NO real IQ2_XXS / IQ2_XS / IQ2_S file is tested (none found publicly 2026-11-16): those are random-block
//! bit-identity to ggml only.
//! Two oracles, per tensor, compared bit for bit (f32 bit patterns):
//! 1. ggml: the file's SHA-256 column is the hash of `dequantize_row_*` output from llama.cpp at
//!    `9d57ce456c`, built by `tools/golden/gguf_dequant_reference.c` (regenerate with
//!    `tools/golden/gen_real_iq_dequant.py`). Our decode must hash the same.
//! 2. Fuel: `fuel-quantized` (crates.io, `GgmlType::to_float` on `BlockIq4Nl` / `BlockIq4Xs` / `BlockIq3S` / `BlockIq3Xxs`).
//!    Its iq4.rs / iq_grid_types.rs / iq_grids.rs at Fuel `origin/main` (0.49.3, unpublished) are
//!    identical to the published 0.47.32 ignoring line endings, measured 2026-11-16.
//!
//! The GGUF parser here is the test's own (header + tensor table), independent of the generator's.

use fuel_quantized::iq_grid_types::{BlockIq3S, BlockIq3Xxs};
use fuel_quantized::iq4::{BlockIq4Nl, BlockIq4Xs};
use fuel_quantized::k_quants::GgmlType;
use ml_convertant_ggml::{
    dequantize_iq3_s, dequantize_iq3_xxs, dequantize_iq4_nl, dequantize_iq4_xs,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    std::env::var_os("MLCONV_MODELS_DIR").map_or_else(|| PathBuf::from("C:/Models"), PathBuf::from)
}

struct Cur<'a>(&'a [u8], usize);
impl Cur<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8_lossy(self.take(n)).into_owned()
    }
    fn skip_value(&mut self, t: u32) {
        match t {
            0 | 1 | 7 => self.1 += 1,
            2 | 3 => self.1 += 2,
            4..=6 => self.1 += 4,
            10..=12 => self.1 += 8,
            8 => {
                self.string();
            }
            9 => {
                let et = self.u32();
                for _ in 0..self.u64() {
                    self.skip_value(et);
                }
            }
            _ => panic!("unknown GGUF value type {t}"),
        }
    }
}

/// (name, ggml type, element count, absolute byte offset) per tensor.
fn gguf_tensors(b: &[u8]) -> Vec<(String, u32, u64, usize)> {
    let mut c = Cur(b, 0);
    assert_eq!(c.take(4), b"GGUF");
    assert_eq!(c.u32(), 3);
    let (nt, nkv) = (c.u64(), c.u64());
    let mut align = 32usize;
    for _ in 0..nkv {
        let k = c.string();
        let t = c.u32();
        if k == "general.alignment" {
            align = c.u32() as usize;
        } else {
            c.skip_value(t);
        }
    }
    let mut ts = Vec::new();
    for _ in 0..nt {
        let name = c.string();
        let nd = c.u32();
        let n: u64 = (0..nd).map(|_| c.u64()).product();
        let ty = c.u32();
        let off = c.u64() as usize;
        ts.push((name, ty, n, off));
    }
    let base = c.1.div_ceil(align) * align;
    ts.into_iter()
        .map(|(n, t, e, o)| (n, t, e, base + o))
        .collect()
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn hash_f32(v: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in v {
        h.update(x.to_le_bytes());
    }
    hex(&h.finalize())
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn fuel_decode<B: GgmlType>(bytes: &[u8], n: usize) -> Vec<f32> {
    let blocks: Vec<B> = bytemuck::pod_collect_to_vec(bytes);
    let mut out = vec![0f32; n];
    B::to_float(&blocks, &mut out);
    out
}

#[test]
fn iq_decode_matches_ggml_and_fuel_on_real_gguf() {
    let tsv = include_str!("golden/real_iq_dequant.tsv");
    let mut golden: HashMap<(String, String), (String, String, usize, String)> = HashMap::new();
    for line in tsv.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 6, "bad golden row {line}");
        golden.insert(
            (f[0].to_string(), f[2].to_string()),
            (
                f[1].to_string(),
                f[3].to_string(),
                f[4].parse().unwrap(),
                f[5].to_string(),
            ),
        );
    }
    let mut files: Vec<&str> = golden.keys().map(|k| k.0.as_str()).collect();
    files.sort_unstable();
    files.dedup();
    assert_eq!(files.len(), 3);
    let mut checked: HashMap<&str, usize> = HashMap::new();
    for file in files {
        let path = models_dir().join("gguf-corpus").join(file);
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("SKIPPED: {} not present", path.display());
            return;
        };
        let want_file = &golden.iter().find(|(k, _)| k.0 == file).unwrap().1.0;
        assert_eq!(
            &hex(&Sha256::digest(&bytes)),
            want_file,
            "{file} is not the pinned file"
        );
        for (name, ty, n, off) in gguf_tensors(&bytes) {
            let (kind, block, blk_bytes) = match ty {
                18 => ("IQ3_XXS", 256, 98),
                20 => ("IQ4_NL", 32, 18),
                21 => ("IQ3_S", 256, 110),
                23 => ("IQ4_XS", 256, 136),
                _ => continue,
            };
            let n = n as usize;
            let raw = &bytes[off..off + n / block * blk_bytes];
            let (_, gk, gn, ghash) = golden
                .get(&(file.to_string(), name.clone()))
                .unwrap_or_else(|| panic!("{file} {name} missing from the golden file"));
            assert_eq!((gk.as_str(), *gn), (kind, n), "{file} {name}");
            let (ours, fuel) = match ty {
                18 => (
                    dequantize_iq3_xxs(raw).unwrap(),
                    fuel_decode::<BlockIq3Xxs>(raw, n),
                ),
                20 => (
                    dequantize_iq4_nl(raw).unwrap(),
                    fuel_decode::<BlockIq4Nl>(raw, n),
                ),
                21 => (
                    dequantize_iq3_s(raw).unwrap(),
                    fuel_decode::<BlockIq3S>(raw, n),
                ),
                _ => (
                    dequantize_iq4_xs(raw).unwrap(),
                    fuel_decode::<BlockIq4Xs>(raw, n),
                ),
            };
            assert_eq!(
                &hash_f32(&ours),
                ghash,
                "{file} {name} ({kind}): ours differs from ggml"
            );
            assert!(
                same_bits(&ours, &fuel),
                "{file} {name} ({kind}): ours differs from Fuel's to_float"
            );
            *checked.entry(kind).or_default() += 1;
        }
    }
    // Every golden row was reached, and every type was covered (control: the counts are the file census).
    assert_eq!(checked.values().sum::<usize>(), golden.len());
    assert_eq!(checked["IQ4_NL"], 540);
    assert_eq!(checked["IQ4_XS"], 30);
    assert_eq!(checked["IQ3_S"], 30);
    assert_eq!(checked["IQ3_XXS"], 30);
    eprintln!("checked {checked:?}");
}
