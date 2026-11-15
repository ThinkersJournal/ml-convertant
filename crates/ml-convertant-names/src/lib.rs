//! Rename LoRA adapter tensors between the PEFT form and the llama.cpp GGUF adapter form.
//!
//! PEFT (`adapter_model.safetensors`): `base_model.model.<module>.lora_A.weight` and `.lora_B.weight`
//! (`.lora_embedding_A` / `.lora_embedding_B`, with no `.weight`, for an embedding).
//! GGUF (`general.type = "adapter"`): `<base tensor name>.lora_a` and `.lora_b`, the base name being the
//! GGUF name of the weight (`blk.0.attn_q.weight`).
//!
//! **Scope.** The Llama-style Hugging Face layout (`model.layers.N.self_attn.{q,k,v,o}_proj`,
//! `model.layers.N.mlp.{gate,up,down}_proj`, `model.embed_tokens`, `lm_head`): Llama, Gemma 2 and Qwen-style
//! models share it. The GGUF side was measured on a real llama adapter (MLMF `lora-measurements.tsv`,
//! `makaveli10/tinyllama-function-call-lora-adapter`); the PEFT side on Llama, Gemma 2, a Qwen-style model, OPT
//! and BLOOM adapters. Anything else (OPT, BLOOM's fused `query_key_value`, MoE experts, ...) is refused with
//! [`NameError::UnknownModule`]: llama.cpp's table at the pinned commit has no entry for OPT's `model.decoder.layers`,
//! and BLOOM's fused projection needs a data permutation that a rename cannot express.
//!
//! **Origin.** The GGUF stems and the rules (`.base_layer.weight` skipped, `lm_head` is `output`, an embedding's
//! `lora_a` transposed) are llama.cpp's `gguf-py/gguf/tensor_mapping.py` and `convert_lora_to_gguf.py` at
//! `9d57ce456c` (MIT), see `THIRD-PARTY-NOTICES.md`.
//!
//! A rename is exact: it carries no numeric error. The one case where the *data* changes with the name is an
//! embedding's `lora_a`, reported by [`Renamed::transpose`]; the caller transposes the bytes.

use std::fmt;

const PEFT_PREFIX: &str = "base_model.model.";

/// Which low-rank factor a tensor holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Factor {
    /// The `A` factor (`lora_A`, GGUF `lora_a`).
    A,
    /// The `B` factor (`lora_B`, GGUF `lora_b`).
    B,
}

/// A tensor name in the other form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Renamed {
    /// The name in the target form.
    pub name: String,
    /// True when the tensor's data must be transposed too (an embedding's `A` factor, both directions).
    pub transpose: bool,
}

/// Why a name was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameError {
    /// A full base weight stored in the adapter (`.base_layer.weight`), not a low-rank factor.
    NotAFactor(String),
    /// A factor of a module this crate has no measured mapping for.
    UnknownModule(String),
    /// Not a PEFT or GGUF adapter tensor name in the expected shape at all.
    Malformed(String),
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAFactor(n) => write!(f, "`{n}` is a base weight, not a LoRA factor"),
            Self::UnknownModule(n) => write!(f, "no measured mapping for the module of `{n}`"),
            Self::Malformed(n) => write!(f, "`{n}` is not a recognised adapter tensor name"),
        }
    }
}

impl std::error::Error for NameError {}

/// (Hugging Face module path below `model.layers.N.`, GGUF stem below `blk.N.`).
const LAYER_MODULES: [(&str, &str); 7] = [
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
];

const EMBED_HF: &str = "model.embed_tokens";
const EMBED_GGUF: &str = "token_embd";
const HEAD_HF: &str = "lm_head";
const HEAD_GGUF: &str = "output";

/// A decimal layer index with no sign, no padding and no overflow, so that the mapping is a bijection.
fn layer_index(s: &str) -> Option<usize> {
    let n: usize = s.parse().ok()?;
    (n.to_string() == s).then_some(n)
}

/// Splits `prefix` + `N` + `.` + rest off `s`.
fn split_layer<'a>(s: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = s.strip_prefix(prefix)?;
    let (idx, tail) = rest.split_once('.')?;
    Some((layer_index(idx)?, tail))
}

/// PEFT name to GGUF name.
///
/// ```
/// use ml_convertant_names::peft_to_gguf;
/// let r = peft_to_gguf("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight").unwrap();
/// assert_eq!(r.name, "blk.3.attn_q.weight.lora_a");
/// assert!(!r.transpose);
/// ```
pub fn peft_to_gguf(name: &str) -> Result<Renamed, NameError> {
    let body = name
        .strip_prefix(PEFT_PREFIX)
        .ok_or_else(|| NameError::Malformed(name.to_owned()))?;
    if body.ends_with(".base_layer.weight") {
        return Err(NameError::NotAFactor(name.to_owned()));
    }
    // (module path, factor, is the embedding spelling)
    let (module, factor, embedding_form) = if let Some(m) = body.strip_suffix(".lora_A.weight") {
        (m, Factor::A, false)
    } else if let Some(m) = body.strip_suffix(".lora_B.weight") {
        (m, Factor::B, false)
    } else if let Some(m) = body.strip_suffix(".lora_embedding_A") {
        (m, Factor::A, true)
    } else if let Some(m) = body.strip_suffix(".lora_embedding_B") {
        (m, Factor::B, true)
    } else {
        return Err(NameError::Malformed(name.to_owned()));
    };
    let unknown = || NameError::UnknownModule(name.to_owned());
    let (stem, is_embed) = if module == EMBED_HF {
        (EMBED_GGUF.to_owned(), true)
    } else if module == HEAD_HF {
        (HEAD_GGUF.to_owned(), false)
    } else if let Some((n, tail)) = split_layer(module, "model.layers.") {
        let (_, g) = LAYER_MODULES
            .iter()
            .find(|(h, _)| *h == tail)
            .ok_or_else(unknown)?;
        (format!("blk.{n}.{g}"), false)
    } else {
        return Err(unknown());
    };
    // `lora_embedding_*` belongs to an embedding module and only to it; a Linear is `lora_A.weight`.
    if embedding_form != is_embed {
        return Err(unknown());
    }
    let suffix = match factor {
        Factor::A => "lora_a",
        Factor::B => "lora_b",
    };
    Ok(Renamed {
        name: format!("{stem}.weight.{suffix}"),
        transpose: is_embed && factor == Factor::A,
    })
}

/// GGUF name to PEFT name.
///
/// ```
/// use ml_convertant_names::gguf_to_peft;
/// let r = gguf_to_peft("token_embd.weight.lora_a").unwrap();
/// assert_eq!(r.name, "base_model.model.model.embed_tokens.lora_embedding_A");
/// assert!(r.transpose);
/// ```
pub fn gguf_to_peft(name: &str) -> Result<Renamed, NameError> {
    let (stem, factor) = if let Some(s) = name.strip_suffix(".weight.lora_a") {
        (s, Factor::A)
    } else if let Some(s) = name.strip_suffix(".weight.lora_b") {
        (s, Factor::B)
    } else {
        return Err(NameError::Malformed(name.to_owned()));
    };
    let unknown = || NameError::UnknownModule(name.to_owned());
    let (module, is_embed) = if stem == EMBED_GGUF {
        (EMBED_HF.to_owned(), true)
    } else if stem == HEAD_GGUF {
        (HEAD_HF.to_owned(), false)
    } else if let Some((n, tail)) = split_layer(stem, "blk.") {
        let (h, _) = LAYER_MODULES
            .iter()
            .find(|(_, g)| *g == tail)
            .ok_or_else(unknown)?;
        (format!("model.layers.{n}.{h}"), false)
    } else {
        return Err(unknown());
    };
    let (letter, weight) = match factor {
        Factor::A => ("A", ".weight"),
        Factor::B => ("B", ".weight"),
    };
    let tail = if is_embed {
        format!("lora_embedding_{letter}")
    } else {
        format!("lora_{letter}{weight}")
    };
    Ok(Renamed {
        name: format!("{PEFT_PREFIX}{module}.{tail}"),
        transpose: is_embed && factor == Factor::A,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names taken from MLMF's `lora-measurements.tsv` (llama, gemma2 and a Qwen-style adapter; the GGUF
    /// column from a real llama adapter), at `origin/main` of mlmf on 2026-11-12.
    const MEASURED: [(&str, &str); 12] = [
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "blk.0.attn_q.weight.lora_a",
        ),
        (
            "base_model.model.model.layers.21.self_attn.k_proj.lora_B.weight",
            "blk.21.attn_k.weight.lora_b",
        ),
        (
            "base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight",
            "blk.1.attn_v.weight.lora_a",
        ),
        (
            "base_model.model.model.layers.2.self_attn.o_proj.lora_B.weight",
            "blk.2.attn_output.weight.lora_b",
        ),
        (
            "base_model.model.model.layers.3.mlp.gate_proj.lora_A.weight",
            "blk.3.ffn_gate.weight.lora_a",
        ),
        (
            "base_model.model.model.layers.4.mlp.up_proj.lora_B.weight",
            "blk.4.ffn_up.weight.lora_b",
        ),
        (
            "base_model.model.model.layers.5.mlp.down_proj.lora_A.weight",
            "blk.5.ffn_down.weight.lora_a",
        ),
        (
            "base_model.model.lm_head.lora_A.weight",
            "output.weight.lora_a",
        ),
        (
            "base_model.model.lm_head.lora_B.weight",
            "output.weight.lora_b",
        ),
        (
            "base_model.model.model.layers.10.mlp.down_proj.lora_B.weight",
            "blk.10.ffn_down.weight.lora_b",
        ),
        (
            "base_model.model.model.embed_tokens.lora_embedding_A",
            "token_embd.weight.lora_a",
        ),
        (
            "base_model.model.model.embed_tokens.lora_embedding_B",
            "token_embd.weight.lora_b",
        ),
    ];

    #[test]
    fn peft_to_gguf_matches_the_measured_names_and_round_trips() {
        for (peft, gguf) in MEASURED {
            let fwd = peft_to_gguf(peft).unwrap();
            assert_eq!(fwd.name, gguf, "{peft}");
            let back = gguf_to_peft(gguf).unwrap();
            assert_eq!(back.name, peft, "{gguf}");
            assert_eq!(fwd.transpose, back.transpose);
        }
    }

    #[test]
    fn only_the_embedding_a_factor_transposes() {
        for (peft, _) in MEASURED {
            let t = peft_to_gguf(peft).unwrap().transpose;
            assert_eq!(t, peft.ends_with("embed_tokens.lora_embedding_A"), "{peft}");
        }
    }

    #[test]
    fn unmeasured_families_are_refused_not_guessed() {
        let opt = "base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight";
        assert_eq!(
            peft_to_gguf(opt),
            Err(NameError::UnknownModule(opt.to_owned()))
        );
        let bloom = "base_model.model.transformer.h.0.self_attention.query_key_value.lora_A.weight";
        assert_eq!(
            peft_to_gguf(bloom),
            Err(NameError::UnknownModule(bloom.to_owned()))
        );
        let moe = "blk.0.ffn_gate_exps.weight.lora_a";
        assert_eq!(
            gguf_to_peft(moe),
            Err(NameError::UnknownModule(moe.to_owned()))
        );
    }

    #[test]
    fn a_base_weight_is_not_a_factor() {
        let n = "base_model.model.lm_head.base_layer.weight";
        assert_eq!(peft_to_gguf(n), Err(NameError::NotAFactor(n.to_owned())));
    }

    #[test]
    fn malformed_names_and_noncanonical_indices_are_refused() {
        for n in [
            "",
            "model.layers.0.self_attn.q_proj.lora_A.weight", // no PEFT prefix
            "base_model.model.model.layers.0.self_attn.q_proj.weight",
            "base_model.model.model.layers.00.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.+1.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers..self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.99999999999999999999999.self_attn.q_proj.lora_A.weight",
        ] {
            assert!(peft_to_gguf(n).is_err(), "{n:?}");
        }
        for n in [
            "",
            "blk.0.attn_q.weight",
            "blk.00.attn_q.weight.lora_a",
            "blk.x.attn_q.weight.lora_a",
        ] {
            assert!(gguf_to_peft(n).is_err(), "{n:?}");
        }
    }

    #[test]
    fn embedding_spelling_belongs_to_the_embedding_only() {
        let n = "base_model.model.model.layers.0.self_attn.q_proj.lora_embedding_A";
        assert!(matches!(peft_to_gguf(n), Err(NameError::UnknownModule(_))));
        let n = "base_model.model.model.embed_tokens.lora_A.weight";
        assert!(matches!(peft_to_gguf(n), Err(NameError::UnknownModule(_))));
    }

    #[test]
    fn every_layer_module_round_trips_for_many_indices() {
        for n in [0usize, 1, 9, 10, 99, 127] {
            for (h, g) in LAYER_MODULES {
                for (peft_suffix, gguf_suffix) in
                    [("lora_A.weight", "lora_a"), ("lora_B.weight", "lora_b")]
                {
                    let peft = format!("base_model.model.model.layers.{n}.{h}.{peft_suffix}");
                    let gguf = format!("blk.{n}.{g}.weight.{gguf_suffix}");
                    assert_eq!(peft_to_gguf(&peft).unwrap().name, gguf);
                    assert_eq!(gguf_to_peft(&gguf).unwrap().name, peft);
                }
            }
        }
    }
}
