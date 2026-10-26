//! GGUF in, GGUF out: MLMF reads and writes the file, `ml-convertant` converts the data.
//!
//! [`convert_gguf`] parses a GGUF with `mlmf-gguf`, converts the tensors a [`Plan`] selects from dense
//! F32/F16/BF16 to a ggml quantized [`Target`], and writes a new GGUF with `mlmf-gguf`'s encoder. Each
//! converted tensor carries an [`ErrorReport`] of the conversion against the source values (see
//! `ml-convertant-core`). Metadata and every tensor the plan does not select are carried over unchanged,
//! including quantized tensors' declared types.
//!
//! The file is refused, not written with gaps, when MLMF could not fully describe it (unrecognized metadata
//! or tensor types, or an incomplete key index): a conversion must not silently drop what it did not
//! understand.
//!
//! Blocks run along the first GGUF dimension (`ne[0]`, the row), so it must be a multiple of the target's
//! block size; other tensors are copied and reported as skipped.

use half::{bf16, f16};
use ml_convertant_core::ErrorReport;
use mlmf_core::{DType, Encoding, MetaValue, MetadataSource, TensorContainer};
use mlmf_ggml::GgmlType;
use mlmf_gguf::{
    GgufEncodeError, GgufError, GgufMetadata, GgufTensorToWrite, GgufVersion, encode_gguf,
    parse_tensors,
};
use std::fmt;

const QUANT_KEY: &str = "general.quantization_version";
/// ggml's `GGML_QNT_VERSION`.
const QUANTIZATION_VERSION: u32 = 2;

/// The ggml quantized type to convert selected tensors to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Target {
    /// Q8_0.
    Q8_0,
    /// Q4_0.
    Q4_0,
    /// Q2_K.
    Q2K,
    /// Q3_K.
    Q3K,
    /// Q4_K.
    Q4K,
    /// Q5_K.
    Q5K,
    /// Q6_K.
    Q6K,
}

impl Target {
    /// Elements per block (or super-block): a tensor's first dimension must be a multiple of this.
    pub const fn block_elems(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q4_0 => 32,
            _ => 256,
        }
    }

    /// The type code MLMF writes for this target.
    pub const fn ggml_type(self) -> GgmlType {
        match self {
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q4_0 => GgmlType::Q4_0,
            Self::Q2K => GgmlType::Q2_K,
            Self::Q3K => GgmlType::Q3_K,
            Self::Q4K => GgmlType::Q4_K,
            Self::Q5K => GgmlType::Q5_K,
            Self::Q6K => GgmlType::Q6_K,
        }
    }

    fn encode(self, x: &[f32]) -> Result<Vec<u8>, String> {
        match self {
            Self::Q8_0 => ml_convertant_ggml::quantize_q8_0(x).map_err(|e| e.to_string()),
            Self::Q4_0 => ml_convertant_ggml::quantize_q4_0(x).map_err(|e| e.to_string()),
            Self::Q2K => ml_convertant_kquants::quantize_q2_k(x).map_err(|e| e.to_string()),
            Self::Q3K => ml_convertant_kquants::quantize_q3_k(x).map_err(|e| e.to_string()),
            Self::Q4K => ml_convertant_kquants::quantize_q4_k(x).map_err(|e| e.to_string()),
            Self::Q5K => ml_convertant_kquants::quantize_q5_k(x).map_err(|e| e.to_string()),
            Self::Q6K => ml_convertant_kquants::quantize_q6_k(x).map_err(|e| e.to_string()),
        }
    }

    fn decode(self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        match self {
            Self::Q8_0 => ml_convertant_ggml::dequantize_q8_0(bytes).map_err(|e| e.to_string()),
            Self::Q4_0 => ml_convertant_ggml::dequantize_q4_0(bytes).map_err(|e| e.to_string()),
            Self::Q2K => ml_convertant_kquants::dequantize_q2_k(bytes).map_err(|e| e.to_string()),
            Self::Q3K => ml_convertant_kquants::dequantize_q3_k(bytes).map_err(|e| e.to_string()),
            Self::Q4K => ml_convertant_kquants::dequantize_q4_k(bytes).map_err(|e| e.to_string()),
            Self::Q5K => ml_convertant_kquants::dequantize_q5_k(bytes).map_err(|e| e.to_string()),
            Self::Q6K => ml_convertant_kquants::dequantize_q6_k(bytes).map_err(|e| e.to_string()),
        }
    }

    fn bound(self, x: &[f32], bytes: &[u8]) -> Result<f64, String> {
        match self {
            Self::Q8_0 => Ok(x
                .chunks(32)
                .map(ml_convertant_ggml::q8_0_block_bound)
                .fold(0.0, f64::max)),
            Self::Q4_0 => Ok(x
                .chunks(32)
                .map(ml_convertant_ggml::q4_0_block_bound)
                .fold(0.0, f64::max)),
            Self::Q2K => ml_convertant_kquants::q2_k_bound(x, bytes).map_err(|e| e.to_string()),
            Self::Q3K => ml_convertant_kquants::q3_k_bound(x, bytes).map_err(|e| e.to_string()),
            Self::Q4K => ml_convertant_kquants::q4_k_bound(x, bytes).map_err(|e| e.to_string()),
            Self::Q5K => ml_convertant_kquants::q5_k_bound(x, bytes).map_err(|e| e.to_string()),
            Self::Q6K => ml_convertant_kquants::q6_k_bound(x, bytes).map_err(|e| e.to_string()),
        }
    }
}

type Select<'p> = Box<dyn Fn(&str, &[usize]) -> bool + 'p>;
/// A tensor ready to write: name, type, dims, bytes.
type Owned = (String, GgmlType, Vec<usize>, Vec<u8>);

/// What to convert: a target type and which tensors.
pub struct Plan<'p> {
    /// The type selected tensors are converted to.
    pub target: Target,
    select: Select<'p>,
}

impl<'p> Plan<'p> {
    /// Selects every tensor of rank two or more (name and shape policy is the caller's: see
    /// [`Plan::with_select`]).
    pub fn new(target: Target) -> Self {
        Self::with_select(target, |_, dims| dims.len() >= 2)
    }

    /// Selects the tensors for which `select(name, dims)` is true; `dims` are GGUF's `ne`, innermost first.
    pub fn with_select(target: Target, select: impl Fn(&str, &[usize]) -> bool + 'p) -> Self {
        Self {
            target,
            select: Box::new(select),
        }
    }
}

/// What happened to one tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Converted from `from` (the source's dense type name) to the plan's target.
    Converted {
        /// The source dense type, `F32`, `F16` or `BF16`.
        from: &'static str,
        /// The conversion's measured error against the source values.
        report: ErrorReport,
    },
    /// Not selected by the plan; copied unchanged.
    Copied,
    /// Selected but not convertible; copied unchanged.
    Skipped {
        /// Why.
        reason: String,
    },
}

/// One tensor's name and what happened to it.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorOutcome {
    /// Tensor name.
    pub name: String,
    /// The outcome.
    pub outcome: Outcome,
}

/// The new file and a per-tensor account.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufConversion {
    /// The complete new GGUF file.
    pub bytes: Vec<u8>,
    /// One entry per tensor, in the input's order.
    pub tensors: Vec<TensorOutcome>,
    /// Metadata keys this crate added because the output needs them and the input did not declare them
    /// (only ever `general.quantization_version`, and only when something was converted).
    pub added_metadata: Vec<String>,
}

/// Why a conversion failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum GlueError {
    /// MLMF could not parse the GGUF.
    Gguf(GgufError),
    /// MLMF reported a failure reading a tensor's bytes.
    Mlmf(mlmf_core::Error),
    /// MLMF could not write the GGUF.
    Encode(GgufEncodeError),
    /// MLMF could not fully describe the file; converting would drop something silently.
    Incomplete(String),
    /// A selected tensor's conversion failed (for example a non-finite value).
    Convert {
        /// Tensor name.
        name: String,
        /// The conversion's message.
        message: String,
    },
    /// A passthrough tensor has a type this build cannot write back.
    Unwritable {
        /// Tensor name.
        name: String,
        /// Why.
        why: String,
    },
}

impl fmt::Display for GlueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf(e) => write!(f, "reading the GGUF: {e}"),
            Self::Mlmf(e) => write!(f, "reading a tensor: {e}"),
            Self::Encode(e) => write!(f, "writing the GGUF: {e:?}"),
            Self::Incomplete(m) => write!(f, "the file cannot be fully carried over: {m}"),
            Self::Convert { name, message } => write!(f, "converting {name}: {message}"),
            Self::Unwritable { name, why } => write!(f, "cannot write {name} back: {why}"),
        }
    }
}

impl std::error::Error for GlueError {}

fn dense_to_f32(dtype: DType, bytes: &[u8]) -> Option<(Vec<f32>, &'static str)> {
    match dtype {
        DType::F32 => Some((
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            "F32",
        )),
        DType::F16 => Some((
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_bits(u16::from_le_bytes(*c)).to_f32())
                .collect(),
            "F16",
        )),
        DType::BF16 => Some((
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
                .collect(),
            "BF16",
        )),
        _ => None,
    }
}

/// GGUF type code for a dense element type this crate can write back.
fn dense_code(dtype: DType) -> Option<u32> {
    match dtype {
        DType::F32 => Some(0),
        DType::F16 => Some(1),
        DType::BF16 => Some(30),
        _ => None,
    }
}

/// Converts the tensors `plan` selects, returning the new GGUF and a per-tensor account.
pub fn convert_gguf(input: &[u8], plan: &Plan<'_>) -> Result<GgufConversion, GlueError> {
    let (meta, meta_report) = GgufMetadata::parse(input, "input").map_err(GlueError::Gguf)?;
    if !meta_report.is_empty() || !meta.index_complete() {
        return Err(GlueError::Incomplete(
            "the metadata block holds values this build of MLMF cannot decode".into(),
        ));
    }
    let (tensors, tensor_report) = parse_tensors(input, &meta, "input").map_err(GlueError::Gguf)?;
    if !tensor_report.is_empty() {
        return Err(GlueError::Incomplete(
            "some tensors have types or ranges this build of MLMF cannot describe".into(),
        ));
    }

    let target = plan.target;
    let block = target.block_elems();
    let mut owned: Vec<Owned> = Vec::new();
    let mut outcomes = Vec::new();
    for d in tensors.tensors() {
        let dims = d.shape.dims().to_vec();
        let raw = tensors.tensor_bytes(d).map_err(GlueError::Mlmf)?;
        let selected = (plan.select)(&d.name, &dims);
        let dense = match d.encoding {
            Encoding::Dense(t) => Some(t),
            _ => None,
        };
        if !selected {
            owned.push(passthrough(&d.name, d.encoding, &dims, &raw)?);
            outcomes.push(TensorOutcome {
                name: d.name.clone(),
                outcome: Outcome::Copied,
            });
            continue;
        }
        let skip = |reason: String,
                    owned: &mut Vec<Owned>,
                    outcomes: &mut Vec<TensorOutcome>|
         -> Result<(), GlueError> {
            owned.push(passthrough(&d.name, d.encoding, &dims, &raw)?);
            outcomes.push(TensorOutcome {
                name: d.name.clone(),
                outcome: Outcome::Skipped { reason },
            });
            Ok(())
        };
        let Some(dtype) = dense.filter(|t| matches!(t, DType::F32 | DType::F16 | DType::BF16))
        else {
            skip(
                "not a dense F32, F16 or BF16 tensor".into(),
                &mut owned,
                &mut outcomes,
            )?;
            continue;
        };
        if dims.first().is_none_or(|&ne0| ne0 % block != 0) {
            skip(
                format!(
                    "first dimension {:?} is not a multiple of the {block}-element block",
                    dims.first()
                ),
                &mut owned,
                &mut outcomes,
            )?;
            continue;
        }
        let (x, from) = dense_to_f32(dtype, &raw).expect("dense type checked above");
        let fail = |message: String| GlueError::Convert {
            name: d.name.clone(),
            message,
        };
        let bytes = target.encode(&x).map_err(&fail)?;
        let decoded = target.decode(&bytes).map_err(&fail)?;
        let bound = target.bound(&x, &bytes).map_err(&fail)?;
        let report = ErrorReport::measure(&x, &decoded)
            .map_err(|e| fail(e.to_string()))?
            .with_bound(bound);
        owned.push((d.name.clone(), target.ggml_type(), dims.clone(), bytes));
        outcomes.push(TensorOutcome {
            name: d.name.clone(),
            outcome: Outcome::Converted { from, report },
        });
    }

    let to_write: Vec<GgufTensorToWrite<'_>> = owned
        .iter()
        .map(|(name, ggml_type, dims, data)| GgufTensorToWrite {
            name,
            ggml_type: *ggml_type,
            dims,
            data,
        })
        .collect();
    let mut entries = meta.entries();
    // GGUF requires `general.quantization_version` once any tensor is quantized, and MLMF supplies no value
    // the file did not declare. Converting is what makes the file quantized, so that value is ours to add:
    // ggml's current quantization version, 2, which llama.cpp's own quantizer writes.
    let quantization_version = MetaValue::U32(QUANTIZATION_VERSION);
    let mut added_metadata = Vec::new();
    let converted_any = outcomes
        .iter()
        .any(|t| matches!(t.outcome, Outcome::Converted { .. }));
    if converted_any && meta.get(QUANT_KEY).is_none() {
        entries.push((QUANT_KEY, &quantization_version));
        added_metadata.push(QUANT_KEY.to_string());
    }
    let bytes = encode_gguf(&entries, &to_write, GgufVersion::V3).map_err(GlueError::Encode)?;
    Ok(GgufConversion {
        bytes,
        tensors: outcomes,
        added_metadata,
    })
}

/// The tensor exactly as declared: its type and bytes.
fn passthrough(
    name: &str,
    encoding: Encoding,
    dims: &[usize],
    raw: &[u8],
) -> Result<Owned, GlueError> {
    let code = match encoding {
        Encoding::Dense(t) => dense_code(t),
        Encoding::Blocked(spec) if spec.family == "ggml" => Some(spec.code),
        _ => None,
    };
    let ty = code
        .and_then(GgmlType::from_code)
        .ok_or_else(|| GlueError::Unwritable {
            name: name.to_string(),
            why: format!("declared encoding {encoding:?} has no GGUF type this build can write"),
        })?;
    Ok((name.to_string(), ty, dims.to_vec(), raw.to_vec()))
}

#[cfg(test)]
mod tests;
