use super::*;
use mlmf_core::MetaValue;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A small GGUF: a 2-D weight (64 x 3), a 1-D norm (64), a 2-D weight whose rows are not block
/// multiples (48 x 2), an F16 weight (64 x 2) and an already-Q8_0 weight (32 x 2).
fn fixture() -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut s = 1u64;
    let w: Vec<f32> = (0..64 * 3).map(|_| lcg(&mut s)).collect();
    let norm: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 0.01).collect();
    let ragged: Vec<f32> = (0..48 * 2).map(|_| lcg(&mut s)).collect();
    let half: Vec<f32> = (0..64 * 2).map(|_| lcg(&mut s)).collect();
    let half_bytes: Vec<u8> = half
        .iter()
        .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let q8_src: Vec<f32> = (0..32 * 2).map(|_| lcg(&mut s)).collect();
    let q8 = ml_convertant_ggml::quantize_q8_0(&q8_src).unwrap();
    let (wb, nb, rb) = (f32_bytes(&w), f32_bytes(&norm), f32_bytes(&ragged));
    let arch = MetaValue::String("llama".to_string());
    let name = MetaValue::String("tiny".to_string());
    let qv = MetaValue::U32(2);
    let tensors = [
        GgufTensorToWrite {
            name: "blk.0.w",
            ggml_type: GgmlType::F32,
            dims: &[64, 3],
            data: &wb,
        },
        GgufTensorToWrite {
            name: "blk.0.norm",
            ggml_type: GgmlType::F32,
            dims: &[64],
            data: &nb,
        },
        GgufTensorToWrite {
            name: "blk.0.ragged",
            ggml_type: GgmlType::F32,
            dims: &[48, 2],
            data: &rb,
        },
        GgufTensorToWrite {
            name: "blk.0.half",
            ggml_type: GgmlType::F16,
            dims: &[64, 2],
            data: &half_bytes,
        },
        GgufTensorToWrite {
            name: "blk.0.q8",
            ggml_type: GgmlType::Q8_0,
            dims: &[32, 2],
            data: &q8,
        },
    ];
    let meta = [
        ("general.architecture", &arch),
        ("general.name", &name),
        ("general.quantization_version", &qv),
    ];
    (
        encode_gguf(&meta, &tensors, GgufVersion::V3).unwrap(),
        w,
        half,
    )
}

fn encodings(bytes: &[u8]) -> Vec<(String, Encoding, Vec<usize>)> {
    let (meta, r) = GgufMetadata::parse(bytes, "t").unwrap();
    assert!(r.is_empty());
    let (t, r) = parse_tensors(bytes, &meta, "t").unwrap();
    assert!(r.is_empty());
    t.tensors()
        .iter()
        .map(|d| (d.name.clone(), d.encoding, d.shape.dims().to_vec()))
        .collect()
}

#[test]
fn converts_selected_dense_tensors_and_carries_the_rest_unchanged() {
    let (input, w, half) = fixture();
    let out = convert_gguf(&input, &Plan::new(Target::Q8_0)).unwrap();
    let outcome = |n: &str| &out.tensors.iter().find(|t| t.name == n).unwrap().outcome;

    let Outcome::Converted { from, report } = outcome("blk.0.w") else {
        panic!("blk.0.w not converted")
    };
    assert_eq!(*from, "F32");
    assert_eq!(report.within_bound(), Some(true));
    let Outcome::Converted { from, .. } = outcome("blk.0.half") else {
        panic!("blk.0.half not converted")
    };
    assert_eq!(*from, "F16");
    assert_eq!(outcome("blk.0.norm"), &Outcome::Copied);
    assert!(matches!(outcome("blk.0.ragged"), Outcome::Skipped { .. }));
    assert!(matches!(outcome("blk.0.q8"), Outcome::Skipped { .. }));

    let list = encodings(&out.bytes);
    let enc = |n: &str| list.iter().find(|t| t.0 == n).unwrap().1;
    let q8_code = GgmlType::Q8_0.code();
    assert!(matches!(enc("blk.0.w"), Encoding::Blocked(s) if s.code == q8_code));
    assert!(matches!(enc("blk.0.half"), Encoding::Blocked(s) if s.code == q8_code));
    assert_eq!(enc("blk.0.norm"), Encoding::Dense(DType::F32));
    assert_eq!(enc("blk.0.ragged"), Encoding::Dense(DType::F32));
    assert!(matches!(enc("blk.0.q8"), Encoding::Blocked(s) if s.code == q8_code));

    let (meta, _) = GgufMetadata::parse(&out.bytes, "t").unwrap();
    let (t, _) = parse_tensors(&out.bytes, &meta, "t").unwrap();
    let bytes_of = |n: &str| t.tensor_bytes(t.tensor(n).unwrap()).unwrap().into_owned();
    assert_eq!(
        bytes_of("blk.0.w"),
        ml_convertant_ggml::quantize_q8_0(&w).unwrap()
    );
    let half_rounded: Vec<f32> = half.iter().map(|x| f16::from_f32(*x).to_f32()).collect();
    assert_eq!(
        bytes_of("blk.0.half"),
        ml_convertant_ggml::quantize_q8_0(&half_rounded).unwrap()
    );

    assert_eq!(
        meta.get("general.architecture"),
        Some(&MetaValue::String("llama".to_string()))
    );
    assert_eq!(
        meta.get("general.name"),
        Some(&MetaValue::String("tiny".to_string()))
    );
}

#[test]
fn a_custom_selection_and_another_target() {
    let (input, _, _) = fixture();
    let plan = Plan::with_select(Target::Q4_0, |name, _| name == "blk.0.w");
    let out = convert_gguf(&input, &plan).unwrap();
    let converted: Vec<&str> = out
        .tensors
        .iter()
        .filter(|t| matches!(t.outcome, Outcome::Converted { .. }))
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(converted, ["blk.0.w"]);
    let list = encodings(&out.bytes);
    let w = list.iter().find(|t| t.0 == "blk.0.w").unwrap();
    assert!(matches!(w.1, Encoding::Blocked(s) if s.code == GgmlType::Q4_0.code()));
    assert_eq!(w.2, vec![64, 3]);
}

#[test]
fn the_report_matches_decoding_the_written_tensor() {
    let (input, w, _) = fixture();
    let out = convert_gguf(&input, &Plan::new(Target::Q8_0)).unwrap();
    let Outcome::Converted { report, .. } = &out.tensors[0].outcome else {
        panic!("first tensor not converted")
    };
    let (meta, _) = GgufMetadata::parse(&out.bytes, "t").unwrap();
    let (t, _) = parse_tensors(&out.bytes, &meta, "t").unwrap();
    let q = t.tensor_bytes(t.tensor("blk.0.w").unwrap()).unwrap();
    let decoded = ml_convertant_ggml::dequantize_q8_0(&q).unwrap();
    let measured = ErrorReport::measure(&w, &decoded).unwrap();
    assert_eq!(report.max_abs, measured.max_abs);
    assert_eq!(report.n, w.len());
}

#[test]
fn a_nonfinite_weight_fails_the_conversion_naming_the_tensor() {
    let mut bad = vec![0.5f32; 64 * 2];
    bad[5] = f32::NAN;
    let bb = f32_bytes(&bad);
    let arch = MetaValue::String("llama".to_string());
    let input = encode_gguf(
        &[("general.architecture", &arch)],
        &[GgufTensorToWrite {
            name: "w",
            ggml_type: GgmlType::F32,
            dims: &[64, 2],
            data: &bb,
        }],
        GgufVersion::V3,
    )
    .unwrap();
    match convert_gguf(&input, &Plan::new(Target::Q8_0)) {
        Err(GlueError::Convert { name, .. }) => assert_eq!(name, "w"),
        other => panic!("expected a conversion error, got {other:?}"),
    }
}

#[test]
fn not_a_gguf_is_an_error() {
    assert!(matches!(
        convert_gguf(b"not a gguf file at all", &Plan::new(Target::Q8_0)),
        Err(GlueError::Gguf(_))
    ));
}

#[test]
fn a_dense_only_file_gains_the_quantization_version_and_says_so() {
    let (arch, bb) = (
        MetaValue::String("llama".to_string()),
        f32_bytes(&[0.25f32; 64]),
    );
    let input = encode_gguf(
        &[("general.architecture", &arch)],
        &[GgufTensorToWrite {
            name: "w",
            ggml_type: GgmlType::F32,
            dims: &[32, 2],
            data: &bb,
        }],
        GgufVersion::V3,
    )
    .unwrap();
    let out = convert_gguf(&input, &Plan::new(Target::Q8_0)).unwrap();
    assert_eq!(out.added_metadata, ["general.quantization_version"]);
    let (meta, _) = GgufMetadata::parse(&out.bytes, "t").unwrap();
    assert_eq!(
        meta.get("general.quantization_version"),
        Some(&MetaValue::U32(2))
    );
    // Nothing converted: nothing added.
    let none = convert_gguf(&input, &Plan::with_select(Target::Q8_0, |_, _| false)).unwrap();
    assert!(none.added_metadata.is_empty());
}
