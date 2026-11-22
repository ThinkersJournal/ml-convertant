//! Real-weight, real-importance-matrix check of the IQ4_NL, IQ4_XS and IQ3_S ENCODERS against ggml's own.
//!
//! Corpus-gated like `real_iq_encode.rs` (`C:/Models`, override `MLCONV_MODELS_DIR`; absent on hosted CI the test
//! prints `SKIPPED` unless `MLCONV_REQUIRE_CORPUS` is set, where a missing or wrong-hash file fails).
//!
//! REAL: 16 consecutive rows each of the q/k/v/o/gate/up/down projections of layers 0, 15 and 29 of
//! SmolLM2-135M-Instruct (Apache-2.0, bf16, `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f7`, pinned by size and SHA-256
//! in `tests/golden/real_checkpoints.tsv`), paired with the importance matrix bartowski published for the same model
//! (`bartowski/SmolLM2-135M-Instruct-GGUF@09816acd`, `SmolLM2-135M-Instruct.imatrix`, legacy llama.cpp format, pinned
//! below). The per-column weight vector is `sum_of_squares / ncall` in f32. Expected hashes (`real_iq_imatrix.tsv`) are
//! the bytes ggml's `quantize_iq*` (llama.cpp `9d57ce456c`) wrote given that vector; ours must be byte-identical.
//! IQ4_XS and IQ3_S need rows of 256 multiples, so they cover ffn_down only (1536 columns); IQ4_NL covers all 21.
//! NOT covered: the imatrix is the file's own value (we did not recompute it from activations), and whether
//! `llama-quantize` divides by `ncall` exactly this way was not re-checked in this test: both sides receive the same vector.
//! Also checks the imatrix is not a no-op: the weighted bytes must differ from the unweighted ones.

use half::bf16;
use ml_convertant_ggml::{
    quantize_iq3_s, quantize_iq3_s_weighted, quantize_iq4_nl, quantize_iq4_nl_weighted,
    quantize_iq4_xs, quantize_iq4_xs_weighted,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

const DIR: &str = "SmolLM2-135M-Instruct";
const IMAT: &str = "imatrix/smollm2-bartowski/SmolLM2-135M-Instruct.imatrix";
const IMAT_SIZE: u64 = 606_008;
const IMAT_SHA: &str = "1248fc3d0ca9bccec848d20f24e93b03a4c738df1d29016824d909301cdecdd7";

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

/// Legacy imatrix: i32 entries; per entry i32 name_len, name, i32 ncall, i32 nval, f32 values[nval]; trailer ignored.
fn read_imatrix(b: &[u8]) -> HashMap<String, Vec<f32>> {
    let i32_at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let mut o = 4;
    let mut out = HashMap::new();
    for _ in 0..i32_at(0) {
        let ln = i32_at(o) as usize;
        o += 4;
        let name = String::from_utf8(b[o..o + ln].to_vec()).unwrap();
        o += ln;
        let ncall = i32_at(o) as f32;
        let nval = i32_at(o + 4) as usize;
        o += 8;
        let w = (0..nval)
            .map(|i| f32::from_le_bytes(b[o + 4 * i..o + 4 * i + 4].try_into().unwrap()) / ncall)
            .collect();
        o += 4 * nval;
        out.insert(name, w);
    }
    out
}

fn sha_file(f: &mut File) -> String {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    hex(&h.finalize())
}

#[test]
fn iq_encode_with_real_imatrix_matches_ggml() {
    let root = std::env::var_os("MLCONV_MODELS_DIR")
        .map_or_else(|| PathBuf::from("C:/Models"), PathBuf::from);
    let (Ok(mut f), Ok(mut fi)) = (
        File::open(root.join(DIR).join("model.safetensors")),
        File::open(root.join(IMAT)),
    ) else {
        return skipped("SmolLM2-135M-Instruct + imatrix");
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
    assert_eq!(sha_file(&mut f), pin[2], "sha256 differs from the pin");
    assert_eq!(
        fi.metadata().unwrap().len(),
        IMAT_SIZE,
        "imatrix size differs from the pin"
    );
    assert_eq!(
        sha_file(&mut fi),
        IMAT_SHA,
        "imatrix sha256 differs from the pin"
    );
    let mut ib = Vec::new();
    fi.seek(SeekFrom::Start(0)).unwrap();
    fi.read_to_end(&mut ib).unwrap();
    let imat = read_imatrix(&ib);
    assert_eq!(imat.len(), 210);

    let mut n = [0u8; 8];
    f.seek(SeekFrom::Start(0)).unwrap();
    f.read_exact(&mut n).unwrap();
    let n = u64::from_le_bytes(n);
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&hdr).unwrap();

    let mut checked: HashMap<String, usize> = HashMap::new();
    for line in include_str!("golden/real_iq_imatrix.tsv").lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let (name, iname, first, rows, ty, want) = (
            c[0],
            c[1],
            c[2].parse::<usize>().unwrap(),
            c[3].parse::<usize>().unwrap(),
            c[4],
            c[5],
        );
        let t = &v[name];
        assert_eq!(t["dtype"], "BF16", "{name}");
        let cols = t["shape"][1].as_u64().unwrap() as usize;
        let off = t["data_offsets"][0].as_u64().unwrap();
        let mut raw = vec![0u8; rows * cols * 2];
        f.seek(SeekFrom::Start(8 + n + off + (first * cols * 2) as u64))
            .unwrap();
        f.read_exact(&mut raw).unwrap();
        let x: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect();
        let w = &imat[iname];
        assert_eq!(w.len(), cols, "{iname}");
        let (got, plain) = match ty {
            "IQ4_NL" => (
                quantize_iq4_nl_weighted(&x, Some(w)).unwrap(),
                quantize_iq4_nl(&x).unwrap(),
            ),
            "IQ4_XS" => (
                quantize_iq4_xs_weighted(&x, Some(w)).unwrap(),
                quantize_iq4_xs(&x).unwrap(),
            ),
            "IQ3_S" => (
                quantize_iq3_s_weighted(&x, Some(w)).unwrap(),
                quantize_iq3_s(&x).unwrap(),
            ),
            _ => panic!("type {ty}"),
        };
        assert_eq!(
            hex(&Sha256::digest(&got)),
            want,
            "{name} rows {first}+{rows} {ty}: weighted encode differs from ggml"
        );
        assert_ne!(got, plain, "{name} {ty}: the imatrix changed nothing");
        *checked.entry(ty.to_string()).or_default() += 1;
    }
    eprintln!("checked {checked:?}");
    assert_eq!(checked["IQ4_NL"], 21);
    assert_eq!(checked["IQ4_XS"], 3);
    assert_eq!(checked["IQ3_S"], 3);
}
