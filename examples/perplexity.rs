//! Perplexity over a text file, computed as llama.cpp's `llama-perplexity`
//! does: the text is tokenized once and cut into chunks of `context`
//! tokens; the first half of each chunk is context, and the log probability
//! of every later token except the first scores the model. The second half
//! is replayed token by token through decode, so each position sees exactly
//! the tokens before it.
//!
//! Usage: `perplexity <model> <text> [chunks=N] [context=N] [prefill=N]
//! [reserve=PERCENT] [bos=ID]`, where `prefill` sets the CUDA or Metal prefill
//! chunk in tokens and `bos` starts the text and replaces each chunk's first
//! token, as `llama-perplexity` does for models whose vocabulary adds a BOS
//! token.
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
    let mut bos = None;
    #[allow(unused_mut)]
    let mut config = RuntimeConfig::default();
    for option in args.filter_map(|option| option.into_string().ok()) {
        if let Some(value) = option.strip_prefix("chunks=") {
            chunks = value.parse()?;
        } else if let Some(value) = option.strip_prefix("bos=") {
            bos = Some(value.parse::<u32>()?);
        } else if let Some(value) = option.strip_prefix("reserve=") {
            // Platform reserve as a percentage of memory (mirmir defaults to
            // 1).
            config.memory.reserve_percent = Some(value.parse()?);
        } else if let Some(value) = option.strip_prefix("tuning=") {
            // `disabled` keeps the heuristic plans, so runs compare without
            // startup measurements choosing different kernels.
            #[cfg(feature = "cuda")]
            {
                config.cuda.tuning.mode = match value {
                    "disabled" => runtime::tuning::TuningMode::Disabled,
                    "startup" => runtime::tuning::TuningMode::Startup,
                    _ => return Err(format!("unknown tuning mode {value}").into()),
                };
            }
            #[cfg(not(feature = "cuda"))]
            let _ = value;
        } else if let Some(value) = option.strip_prefix("context=") {
            context = value.parse()?;
        } else if let Some(value) = option.strip_prefix("prefill=") {
            #[cfg(feature = "cuda")]
            {
                config.cuda.model_session.prefill_chunk_tokens = value.parse()?;
            }
            #[cfg(feature = "metal")]
            {
                config.metal.cache.prefill_step = Some(value.parse()?);
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            let _ = value;
        } else {
            return Err(format!("unknown option {option}").into());
        }
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
    let mut tokens = tokenizer.encode(fs::read_to_string(text)?, false)?.get_ids().to_vec();
    // `llama-perplexity` tokenizes with the BOS token in front of the text.
    if let Some(bos) = bos {
        tokens.insert(0, bos);
    }
    let chunks = chunks.min(tokens.len() / context);
    let first = context / 2;
    let library = Library::new(config);
    let model = library.load(model_path, GenerationOverrides::default(), &mut |_| {})?;
    let (mut total, mut count) = (0.0_f64, 0_u32);
    let mut stdout = std::io::stdout().lock();
    for chunk in tokens.chunks_exact(context).take(chunks) {
        let mut chunk = chunk.to_vec();
        if let (Some(bos), Some(first)) = (bos, chunk.first_mut()) {
            *first = bos;
        }
        let chunk = chunk.as_slice();
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
