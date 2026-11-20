//! LoRA merge against weights merged by PEFT itself (peft 0.22.0, `merge_and_unload`).
//!
//! Corpus-gated like `real_iq_encode.rs`: the base (Qwen3-0.6B, Apache-2.0, layer 0 `self_attn.v_proj`, bf16 widened
//! to f32) lives in `C:/Models` (override `MLCONV_MODELS_DIR`); absent on hosted CI the test prints `SKIPPED` unless
//! `MLCONV_REQUIRE_CORPUS` is set, where a missing or wrong-hash file fails. Pinned in `real_checkpoints.tsv`.
//!
//! REAL: PEFT's merged weights (first 32 rows) for two adapters, standard LoRA (rank 4, alpha 8) and rsLoRA
//! (rank 8, alpha 16). SYNTHETIC: the adapter values (seeded random A and B written by `tools/golden/gen_peft_merge.py`,
//! which also records the peft/transformers/torch versions in `tests/golden/peft_merge/manifest.tsv`); no trained
//! adapter for this base is staged. Not covered: other target modules, dtype other than f32, `use_dora`, `lora_dropout`.
//! PEFT sums in torch's f32 matmul order, ours in a fixed ascending order, so the two are compared within a bound,
//! not bit for bit: `|ours - PEFT| <= 2 * lora_merge_bound` (our proven bound against the exact result, twice).

use half::bf16;
use ml_convertant_lora::{LoraAdapter, lora_merge_bound, lora_scale, merge_lora, rslora_scale};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

const DIR: &str = "Qwen3-0.6B-dense";
const TENSOR: &str = "model.layers.0.self_attn.v_proj.weight";

fn skipped(what: &str) {
    assert!(
        std::env::var_os("MLCONV_REQUIRE_CORPUS").is_none(),
        "corpus required but missing: {what}"
    );
    eprintln!("SKIPPED: {what}");
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn lora_merge_matches_peft() {
    let dir = std::env::var_os("MLCONV_MODELS_DIR")
        .map_or_else(|| PathBuf::from("C:/Models"), PathBuf::from)
        .join(DIR);
    let Ok(mut f) = File::open(dir.join("model.safetensors")) else {
        return skipped(DIR);
    };
    let pin = include_str!("golden/real_checkpoints.tsv")
        .lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .find(|r| r[0] == DIR)
        .expect("pin row");
    assert_eq!(
        f.metadata().unwrap().len().to_string(),
        pin[1],
        "size differs from the pin"
    );
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, pin[2], "sha256 differs from the pin");

    f.seek(SeekFrom::Start(0)).unwrap();
    let mut n = [0u8; 8];
    f.read_exact(&mut n).unwrap();
    let n = u64::from_le_bytes(n);
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&hdr).unwrap();
    let t = &v[TENSOR];
    assert_eq!(t["dtype"], "BF16");
    let (out, inp) = (
        t["shape"][0].as_u64().unwrap() as usize,
        t["shape"][1].as_u64().unwrap() as usize,
    );
    let off = t["data_offsets"][0].as_u64().unwrap();

    let ab = [
        (
            "lora_r4",
            &include_bytes!("golden/peft_merge/lora_r4_ab.bin")[..],
            &include_bytes!("golden/peft_merge/lora_r4_merged.bin")[..],
        ),
        (
            "rslora_r8",
            &include_bytes!("golden/peft_merge/rslora_r8_ab.bin")[..],
            &include_bytes!("golden/peft_merge/rslora_r8_merged.bin")[..],
        ),
    ];
    let mut checked = 0;
    for line in include_str!("golden/peft_merge/manifest.tsv").lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (case, rank, alpha, rs, m_out, m_in, rows) = (
            c[0],
            c[1].parse::<usize>().unwrap(),
            c[2].parse::<f32>().unwrap(),
            c[3] == "1",
            c[4].parse::<usize>().unwrap(),
            c[5].parse::<usize>().unwrap(),
            c[6].parse::<usize>().unwrap(),
        );
        assert_eq!((m_out, m_in), (out, inp), "{case}: shape");
        let (_, abb, mb) = ab.iter().find(|x| x.0 == case).expect(case);
        let (a, b) = abb.split_at(rank * inp * 4);
        let (a, b) = (f32s(a), f32s(b));
        assert_eq!(b.len(), out * rank, "{case}: B len");
        let peft = f32s(mb);
        assert_eq!(peft.len(), rows * inp);

        // Base rows 0..rows, bf16 widened to f32 (exact), as the generator loaded them.
        let mut raw = vec![0u8; rows * inp * 2];
        f.seek(SeekFrom::Start(8 + n + off)).unwrap();
        f.read_exact(&mut raw).unwrap();
        let base: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| bf16::from_bits(u16::from_le_bytes(*x)).to_f32())
            .collect();

        let scale = if rs {
            rslora_scale(alpha, rank)
        } else {
            lora_scale(alpha, rank)
        };
        // Rows are independent: merge the first `rows` rows with the matching rows of B.
        let ad = LoraAdapter {
            out_features: rows,
            in_features: inp,
            rank,
            a,
            b: b[..rows * rank].to_vec(),
            scale,
        };
        let ours = merge_lora(&base, &ad).unwrap();
        let bound = lora_merge_bound(&base, &ad).unwrap();
        let (mut worst, mut changed) = (0f64, 0usize);
        for ((o, p), w) in ours.iter().zip(&peft).zip(&base) {
            worst = worst.max((f64::from(*o) - f64::from(*p)).abs());
            changed += usize::from(o != w);
        }
        let exact = ours.iter().zip(&peft).filter(|(o, p)| o == p).count();
        eprintln!(
            "{case}: {rows}x{inp} scale {scale}: max |ours-PEFT| {worst:e}, bound {bound:e}, bit-equal {exact}/{}, merge changed {changed}",
            ours.len()
        );
        assert!(
            changed > ours.len() / 2,
            "{case}: the merge barely changed the base; the test would not detect a missing adapter"
        );
        assert!(worst <= 2.0 * bound, "{case}: {worst:e} > 2 * {bound:e}");
        checked += 1;
    }
    assert_eq!(checked, 2);
}
