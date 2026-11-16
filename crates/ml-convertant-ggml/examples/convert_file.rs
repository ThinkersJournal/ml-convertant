//! `convert_file <q8_0|q4_0|iq4_nl> <in.f32> <out.bin> <out.deq.f32> [importance.f32]`: encodes a raw little-endian f32 file
//! and writes the blocks and their decoding, in the same file formats as `tools/golden/*_reference.c`,
//! so the CI differential can compare the two byte for byte.

use ml_convertant_ggml::{
    dequantize_iq4_nl, dequantize_q4_0, dequantize_q8_0, quantize_iq4_nl_weighted, quantize_q4_0,
    quantize_q8_0,
};
use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let (format, input, out_bin, out_deq, importance) = match args.as_slice() {
        [_, f, i, b, d] => (f, i, b, d, None),
        [_, f, i, b, d, w] => (f, i, b, d, Some(w)),
        _ => {
            return Err(
                "usage: convert_file <q8_0|q4_0|iq4_nl> <in.f32> <out.bin> <out.deq.f32> [importance.f32]"
                    .into(),
            );
        }
    };
    let read = |path: &String| -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(fs::read(path)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let x = read(input)?;
    let importance = importance.map(read).transpose()?;
    let (blocks, decoded) = match format.as_str() {
        "q8_0" => {
            let b = quantize_q8_0(&x)?;
            let d = dequantize_q8_0(&b)?;
            (b, d)
        }
        "q4_0" => {
            let b = quantize_q4_0(&x)?;
            let d = dequantize_q4_0(&b)?;
            (b, d)
        }
        "iq4_nl" => {
            let b = quantize_iq4_nl_weighted(&x, importance.as_deref())?;
            let d = dequantize_iq4_nl(&b)?;
            (b, d)
        }
        other => return Err(format!("unknown format {other}").into()),
    };
    fs::write(out_bin, blocks)?;
    fs::write(
        out_deq,
        decoded
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    Ok(())
}
