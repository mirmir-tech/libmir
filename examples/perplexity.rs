//! Perplexity over a text file, computed as llama.cpp's `llama-perplexity`
//! does: the text is tokenized once and cut into chunks of `context`
//! tokens; the first half of each chunk is context, and the log probability
//! of every later token except the first scores the model. The second half
//! is replayed token by token through decode, so each position sees exactly
//! the tokens before it.
//!
//! Usage: `perplexity <model> <text> [chunks=N] [context=N] [prefill=N]`,
//! where `prefill` sets the CUDA prefill chunk in tokens.
#![allow(clippy::print_stdout)]

use std::{env, fs, io::Write, path::PathBuf};

use libmir::{GenerationOverrides, Library, RuntimeConfig, SamplingLogits};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = env::args_os().skip(1);
    let model_path = args.next().map(PathBuf::from).ok_or("missing model path")?;
    let text = args.next().map(PathBuf::from).ok_or("missing text path")?;
    let mut chunks = usize::MAX;
    let mut context = 512_usize;
    #[allow(unused_mut)]
    let mut config = RuntimeConfig::default();
    for option in args.filter_map(|option| option.into_string().ok()) {
        if let Some(value) = option.strip_prefix("chunks=") {
            chunks = value.parse()?;
        } else if let Some(value) = option.strip_prefix("context=") {
            context = value.parse()?;
        } else if let Some(value) = option.strip_prefix("prefill=") {
            #[cfg(feature = "cuda")]
            {
                config.cuda.model_session.prefill_chunk_tokens = value.parse()?;
            }
            #[cfg(not(feature = "cuda"))]
            let _ = value;
        } else {
            return Err(format!("unknown option {option}").into());
        }
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
    let tokens = tokenizer.encode(fs::read_to_string(text)?, false)?.get_ids().to_vec();
    let chunks = chunks.min(tokens.len() / context);
    let first = context / 2;
    let library = Library::new(config);
    let model = library.load(model_path, GenerationOverrides::default(), &mut |_| {})?;
    let (mut total, mut count) = (0.0_f64, 0_u32);
    let mut stdout = std::io::stdout().lock();
    for chunk in tokens.chunks_exact(context).take(chunks) {
        let mut session = model.session();
        let output = session.prefill(&chunk[..=first], SamplingLogits::Full, &mut |_| {})?;
        let mut logits = output.logits.ok_or("prefill returned no logits")?.values;
        for (position, &token) in chunk.iter().enumerate().skip(first + 1) {
            total -= log_probability(&logits, token)?;
            count += 1;
            if position + 1 < context {
                logits = session
                    .decode(token, SamplingLogits::Full)?
                    .logits
                    .ok_or("decode returned no logits")?
                    .values;
            }
        }
        write!(
            stdout,
            "[{}]{:.4} ",
            count / u32::try_from(context - first - 1)?,
            (total / f64::from(count)).exp()
        )?;
        stdout.flush()?;
    }
    writeln!(
        stdout,
        "\nchunks={chunks} context={context} tokens={count} perplexity={:.4}",
        (total / f64::from(count)).exp()
    )?;
    model.unload()?;
    Ok(())
}

fn log_probability(
    logits: &[f32],
    token: u32,
) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|value| f64::from(value - maximum).exp()).sum::<f64>();
    let value = logits.get(usize::try_from(token)?).ok_or("token outside the vocabulary")?;
    Ok(f64::from(value - maximum) - sum.ln())
}
