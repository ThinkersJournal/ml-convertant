//! Safetensors in, GGUF out: MLMF reads the file, `ml-convertant` converts the data, MLMF writes the GGUF.
//!
//! [`convert_safetensors`] is [`crate::convert_gguf`] for a safetensors source. The per-tensor rules, the
//! [`Plan`], the [`ErrorReport`](ml_convertant_core::ErrorReport) and the outcomes are the same ones.
//!
//! What differs, because safetensors says less than GGUF:
//! - Shapes are stored outermost first and GGUF's `ne` is innermost first, so the dimensions are reversed;
//!   the bytes are row-major in both and are not moved. A [`Plan`] selector sees the GGUF `ne`.
//! - A scalar (rank 0) is written as a one-element vector, which GGUF can describe.
//! - Safetensors names are carried over exactly as they are (HuggingFace spellings): translating them to
//!   GGUF names is MLMF's `mlmf-tensor-names`, and which names a model needs is the consumer's call.
//! - Safetensors' `__metadata__` strings are not carried over: GGUF has its own metadata vocabulary
//!   (`general.architecture` and the rest), which the caller supplies as `metadata`.
//!
//! Like the GGUF path, the file is refused rather than written with gaps when MLMF could not describe all
//! of it, and a tensor this build cannot write into a GGUF (for example an integer tensor) is an error
//! rather than a silent drop.

use crate::{GgufConversion, GlueError, Owned, Plan, TensorOutcome, convert_one, write_gguf};
use mlmf_core::{MetaValue, TensorContainer};

/// Converts the tensors `plan` selects from a safetensors file image to a GGUF.
///
/// `metadata` is written to the GGUF as given (`general.architecture`, tokenizer keys, ...);
/// `general.quantization_version` is added when something was converted and `metadata` lacks it.
///
/// # Errors
///
/// See [`GlueError`]: `Safetensors` when the container does not parse, `Incomplete` when MLMF could not
/// describe every tensor, `Convert` when a selected tensor's conversion fails, `Unwritable` for a tensor
/// whose element type has no GGUF spelling here.
pub fn convert_safetensors(
    input: &[u8],
    plan: &Plan<'_>,
    metadata: &[(&str, &MetaValue)],
) -> Result<GgufConversion, GlueError> {
    let header = mlmf_safetensors::parse_header(input).map_err(GlueError::Safetensors)?;
    let (tensors, report) =
        mlmf_safetensors::parse_tensors(input, &header, "input").map_err(GlueError::Safetensors)?;
    if !report.is_empty() {
        return Err(GlueError::Incomplete(
            "some tensors have types or ranges this build of MLMF cannot describe".into(),
        ));
    }
    let mut owned: Vec<Owned> = Vec::new();
    let mut outcomes: Vec<TensorOutcome> = Vec::new();
    for d in tensors.tensors() {
        let mut dims = d.shape.dims().to_vec();
        dims.reverse();
        if dims.is_empty() {
            dims.push(1);
        }
        let raw = tensors.tensor_bytes(d).map_err(GlueError::Mlmf)?;
        let (o, outcome) = convert_one(plan, &d.name, dims, d.encoding, &raw)?;
        owned.push(o);
        outcomes.push(outcome);
    }
    let declares = metadata.iter().any(|(k, _)| *k == crate::QUANT_KEY);
    write_gguf(metadata, owned, outcomes, declares)
}

#[cfg(test)]
mod tests;
