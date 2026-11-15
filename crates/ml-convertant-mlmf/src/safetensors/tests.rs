use super::*;
use crate::{Outcome, Target};
use half::bf16;
use mlmf_core::{DType, Encoding, MetadataSource};
use mlmf_ggml::GgmlType;
use mlmf_gguf::{GgufMetadata, parse_tensors};
use mlmf_safetensors::{TensorToWrite, encode_safetensors};

const NO_META: Option<&[(&str, &str)]> = None;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Fixture {
    file: Vec<u8>,
    w: Vec<f32>,
    b: Vec<f32>,
}

/// `w`: F32 [3, 64] (3 rows of 64), `b`: BF16 [2, 64], `norm`: F32 [64], `ragged`: F32 [2, 48],
/// `scale`: F32 scalar.
fn fixture() -> Fixture {
    let mut s = 7u64;
    let w: Vec<f32> = (0..3 * 64).map(|_| lcg(&mut s)).collect();
    let b: Vec<f32> = (0..2 * 64).map(|_| lcg(&mut s)).collect();
    let b_bytes: Vec<u8> = b
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let norm: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 0.01).collect();
    let ragged: Vec<f32> = (0..2 * 48).map(|_| lcg(&mut s)).collect();
    let (wb, nb, rb, sb) = (
        f32_bytes(&w),
        f32_bytes(&norm),
        f32_bytes(&ragged),
        f32_bytes(&[0.5]),
    );
    let t = |name, dtype, shape: &'static [usize], data: &'static [u8]| TensorToWrite {
        name,
        dtype,
        shape,
        data,
    };
    // Leak the small buffers so the borrows are 'static for the helper above.
    let leak = |v: Vec<u8>| -> &'static [u8] { Box::leak(v.into_boxed_slice()) };
    let tensors = [
        t("w", DType::F32, &[3, 64], leak(wb)),
        t("b", DType::BF16, &[2, 64], leak(b_bytes)),
        t("norm", DType::F32, &[64], leak(nb)),
        t("ragged", DType::F32, &[2, 48], leak(rb)),
        t("scale", DType::F32, &[], leak(sb)),
    ];
    Fixture {
        file: encode_safetensors(&tensors, NO_META).unwrap(),
        w,
        b,
    }
}

type Row = (String, Encoding, Vec<usize>, Vec<u8>);

fn gguf_tensors(bytes: &[u8]) -> (GgufMetadata<'_>, Vec<Row>) {
    let (meta, r) = GgufMetadata::parse(bytes, "t").unwrap();
    assert!(r.is_empty());
    let (t, r) = parse_tensors(bytes, &meta, "t").unwrap();
    assert!(r.is_empty());
    let list = t
        .tensors()
        .iter()
        .map(|d| {
            (
                d.name.clone(),
                d.encoding,
                d.shape.dims().to_vec(),
                t.tensor_bytes(d).unwrap().into_owned(),
            )
        })
        .collect();
    (meta, list)
}

#[test]
fn quantizes_dense_tensors_reverses_dims_and_keeps_the_row_major_bytes() {
    let fx = fixture();
    let arch = MetaValue::String("llama".to_string());
    let out = convert_safetensors(
        &fx.file,
        &Plan::new(Target::Q8_0),
        &[("general.architecture", &arch)],
    )
    .unwrap();
    let (meta, list) = gguf_tensors(&out.bytes);
    let get = |n: &str| list.iter().find(|t| t.0 == n).unwrap();
    let q8 = GgmlType::Q8_0.code();

    // [3, 64] in safetensors is ne = [64, 3] in GGUF, and Q8_0 blocks run along the 64.
    let w = get("w");
    assert_eq!(w.2, vec![64, 3]);
    assert!(matches!(w.1, Encoding::Blocked(s) if s.code == q8));
    assert_eq!(w.3, ml_convertant_ggml::quantize_q8_0(&fx.w).unwrap());

    let b = get("b");
    assert_eq!(b.2, vec![64, 2]);
    let b_rounded: Vec<f32> = fx.b.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
    assert_eq!(b.3, ml_convertant_ggml::quantize_q8_0(&b_rounded).unwrap());

    assert_eq!(
        meta.get("general.architecture"),
        Some(&MetaValue::String("llama".to_string()))
    );
    assert_eq!(out.added_metadata, vec!["general.quantization_version"]);
    assert_eq!(
        meta.get("general.quantization_version"),
        Some(&MetaValue::U32(2))
    );
}

#[test]
fn reports_per_tensor_and_leaves_vectors_ragged_rows_and_scalars_alone() {
    let fx = fixture();
    let arch = MetaValue::String("llama".to_string());
    let out = convert_safetensors(
        &fx.file,
        &Plan::new(Target::Q8_0),
        &[("general.architecture", &arch)],
    )
    .unwrap();
    let outcome = |n: &str| &out.tensors.iter().find(|t| t.name == n).unwrap().outcome;

    let Outcome::Converted { from, report } = outcome("w") else {
        panic!("w not converted")
    };
    assert_eq!(*from, "F32");
    assert_eq!(report.within_bound(), Some(true));
    let Outcome::Converted { from, report } = outcome("b") else {
        panic!("b not converted")
    };
    assert_eq!(*from, "BF16");
    assert_eq!(report.within_bound(), Some(true));
    assert_eq!(outcome("norm"), &Outcome::Copied);
    assert_eq!(outcome("scale"), &Outcome::Copied);
    assert!(matches!(outcome("ragged"), Outcome::Skipped { .. }));

    let (_, list) = gguf_tensors(&out.bytes);
    let get = |n: &str| list.iter().find(|t| t.0 == n).unwrap();
    assert_eq!(get("norm").1, Encoding::Dense(DType::F32));
    assert_eq!(get("ragged").2, vec![48, 2]);
    // The scalar became a one-element vector with its value intact.
    assert_eq!(get("scale").2, vec![1]);
    assert_eq!(get("scale").3, 0.5f32.to_le_bytes().to_vec());
}

#[test]
fn a_selector_sees_the_gguf_ne_and_a_declared_quantization_version_is_kept() {
    let fx = fixture();
    let seen = std::cell::RefCell::new(Vec::new());
    let arch = MetaValue::String("llama".to_string());
    let qv = MetaValue::U32(2);
    let plan = Plan::with_select(Target::Q4_0, |name, dims| {
        seen.borrow_mut().push((name.to_string(), dims.to_vec()));
        name == "w"
    });
    let meta = [
        ("general.architecture", &arch),
        ("general.quantization_version", &qv),
    ];
    let out = convert_safetensors(&fx.file, &plan, &meta).unwrap();
    assert!(out.added_metadata.is_empty());
    assert!(
        seen.borrow()
            .iter()
            .any(|(n, d)| n == "w" && *d == vec![64usize, 3])
    );
    let outcome = |n: &str| &out.tensors.iter().find(|t| t.name == n).unwrap().outcome;
    assert!(matches!(outcome("w"), Outcome::Converted { .. }));
    assert_eq!(outcome("b"), &Outcome::Copied);
}

#[test]
fn a_file_that_is_not_safetensors_is_an_error_not_an_empty_conversion() {
    let err =
        convert_safetensors(b"not a safetensors file", &Plan::new(Target::Q8_0), &[]).unwrap_err();
    assert!(matches!(err, GlueError::Safetensors(_)), "{err}");
}

#[test]
fn an_integer_tensor_is_refused_by_name() {
    let ids = [1u8, 0, 0, 0, 0, 0, 0, 0];
    let t = TensorToWrite {
        name: "ids",
        dtype: DType::I64,
        shape: &[1],
        data: &ids,
    };
    let file = encode_safetensors(&[t], NO_META).unwrap();
    let err = convert_safetensors(&file, &Plan::new(Target::Q8_0), &[]).unwrap_err();
    assert!(
        matches!(&err, GlueError::Unwritable { name, .. } if name == "ids"),
        "{err}"
    );
}

#[test]
fn metadata_without_the_architecture_is_the_encoders_refusal() {
    let fx = fixture();
    let err = convert_safetensors(&fx.file, &Plan::new(Target::Q8_0), &[]).unwrap_err();
    assert!(
        matches!(&err, GlueError::Encode(e) if format!("{e:?}").contains("general.architecture")),
        "{err}"
    );
}
