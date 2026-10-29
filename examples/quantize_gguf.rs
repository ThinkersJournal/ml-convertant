//! `cargo run --release --example quantize_gguf -- in.gguf out.gguf [q8_0|q4_0]`
use ml_convertant::mlmf::{Outcome, Plan, Target, convert_gguf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let target = if args.get(3).map(String::as_str) == Some("q4_0") {
        Target::Q4_0
    } else {
        Target::Q8_0
    };
    let out = convert_gguf(&std::fs::read(&args[1])?, &Plan::new(target))?;
    std::fs::write(&args[2], &out.bytes)?;
    for t in &out.tensors {
        if let Outcome::Converted { report, .. } = &t.outcome {
            println!(
                "{}: max error {:.3e}, rmse {:.3e}",
                t.name, report.max_abs, report.rmse
            );
        }
    }
    Ok(())
}
