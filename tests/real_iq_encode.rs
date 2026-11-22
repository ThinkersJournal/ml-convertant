//! Real-weight check of the IQ4_NL, IQ4_XS and IQ3_S ENCODERS against ggml's own.
//!
//! Corpus-gated like `real_iq_files.rs`: the checkpoint lives in `C:/Models` (override with `MLCONV_MODELS_DIR`);
//! absent on hosted CI the test prints `SKIPPED` unless `MLCONV_REQUIRE_CORPUS` is set (the corpus job), where a
//! missing or wrong-hash file fails.
//!
//! REAL: 16 consecutive rows each of 16 tensors of Qwen3-0.6B (Apache-2.0, bf16, pinned by size and SHA-256 in
//! `tests/golden/real_checkpoints.tsv`): q/k/o/gate/down projections of layers 0, 13 and 27, plus the embedding matrix.
//! Selection is in `tests/golden/real_iq_encode.tsv` (tensor, first row, rows, type, SHA-256 of the bytes ggml's
//! `quantize_iq*` wrote, from llama.cpp `9d57ce456c`; regenerate with `tools/golden/gen_real_iq_encode.py`).
//! imatrix encoding on real weights is `real_iq_imatrix.rs` (SmolLM2-135M); not covered here.
//! Our encode of the same f32 values must hash identically (byte-identical output).

use half::bf16;
use ml_convertant_ggml::{quantize_iq3_s, quantize_iq4_nl, quantize_iq4_xs};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

const DIR: &str = "Qwen3-0.6B-dense";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn skipped(what: &str) {
    assert!(
        std::env::var_os("MLCONV_REQUIRE_CORPUS").is_none(),
        "corpus required but missing: {what}"
    );
    eprintln!("SKIPPED: {what}");
}

#[test]
fn iq_encode_matches_ggml_on_real_weights() {
    let dir = std::env::var_os("MLCONV_MODELS_DIR")
        .map_or_else(|| PathBuf::from("C:/Models"), PathBuf::from)
        .join(DIR);
    let Ok(mut f) = File::open(dir.join("model.safetensors")) else {
        return skipped(DIR);
    };
    // pin: size and SHA-256 from real_checkpoints.tsv
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
    assert_eq!(hex(&h.finalize()), pin[2], "sha256 differs from the pin");

    let mut n = [0u8; 8];
    f.seek(SeekFrom::Start(0)).unwrap();
    f.read_exact(&mut n).unwrap();
    let n = u64::from_le_bytes(n);
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&hdr).unwrap();

    let mut cache: HashMap<(String, usize), Vec<f32>> = HashMap::new();
    let mut checked: HashMap<String, usize> = HashMap::new();
    for line in include_str!("golden/real_iq_encode.tsv").lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (name, first, rows, ty, want) = (
            c[0],
            c[1].parse::<usize>().unwrap(),
            c[2].parse::<usize>().unwrap(),
            c[3],
            c[4],
        );
        let x = cache.entry((name.to_string(), first)).or_insert_with(|| {
            let t = &v[name];
            assert_eq!(t["dtype"], "BF16", "{name}");
            let cols = t["shape"][1].as_u64().unwrap() as usize;
            let off = t["data_offsets"][0].as_u64().unwrap();
            let mut raw = vec![0u8; rows * cols * 2];
            f.seek(SeekFrom::Start(8 + n + off + (first * cols * 2) as u64))
                .unwrap();
            f.read_exact(&mut raw).unwrap();
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect()
        });
        let got = match ty {
            "IQ4_NL" => quantize_iq4_nl(x).unwrap(),
            "IQ4_XS" => quantize_iq4_xs(x).unwrap(),
            "IQ3_S" => quantize_iq3_s(x).unwrap(),
            _ => panic!("type {ty}"),
        };
        assert_eq!(
            hex(&Sha256::digest(&got)),
            want,
            "{name} rows {first}+{rows} {ty}: encode differs from ggml"
        );
        *checked.entry(ty.to_string()).or_default() += 1;
    }
    eprintln!("checked {checked:?}");
    assert_eq!(checked.len(), 3);
    assert!(checked.values().all(|&n| n == 16), "{checked:?}");
}
