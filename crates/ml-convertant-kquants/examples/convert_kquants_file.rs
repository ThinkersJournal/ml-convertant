//! `convert_kquants_file <q2_k|q3_k|q4_k|q5_k|q6_k> <in.f32> <out.bin> <out.deq.f32>`: encodes a raw little-endian f32 file
//! (a multiple of 256 values) and writes the super-blocks and their decoding, in the same file formats as
//! `tools/golden/*_reference.c`, so the CI differential can compare the two byte for byte.

use ml_convertant_kquants::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k,
    quantize_q2_k, quantize_q3_k, quantize_q4_k, quantize_q5_k, quantize_q6_k,
};
use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let [_, format, input, out_bin, out_deq] = args.as_slice() else {
        return Err(
            "usage: convert_kquants_file <q2_k|q3_k|q4_k|q5_k|q6_k> <in.f32> <out.bin> <out.deq.f32>".into(),
        );
    };
    let x: Vec<f32> = fs::read(input)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let (blocks, decoded) = match format.as_str() {
        "q2_k" => {
            let b = quantize_q2_k(&x)?;
            let d = dequantize_q2_k(&b)?;
            (b, d)
        }
        "q3_k" => {
            let b = quantize_q3_k(&x)?;
            let d = dequantize_q3_k(&b)?;
            (b, d)
        }
        "q4_k" => {
            let b = quantize_q4_k(&x)?;
            let d = dequantize_q4_k(&b)?;
            (b, d)
        }
        "q5_k" => {
            let b = quantize_q5_k(&x)?;
            let d = dequantize_q5_k(&b)?;
            (b, d)
        }
        "q6_k" => {
            let b = quantize_q6_k(&x)?;
            let d = dequantize_q6_k(&b)?;
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
