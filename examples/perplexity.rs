//! Perplexity over a text file, computed as llama.cpp's `llama-perplexity`
//! does: the text is tokenized once and cut into chunks of `context`
//! tokens; the first half of each chunk is context, and the log probability
//! of every later token except the first scores the model. The second half
//! is replayed token by token through decode, so each position sees exactly
//! the tokens before it.
//!
//! Usage: `perplexity <model> <text> [chunks=N] [context=N] [prefill=N]
//! [reserve=PERCENT] [bos=ID|none]`, where `prefill` sets the CUDA or Metal
//! prefill chunk in tokens. A BOS token starts the text and replaces each
//! chunk's first token, as `llama-perplexity` does for models whose
//! vocabulary adds one: by default an explicit `add_bos_token` in
//! `tokenizer_config.json` decides, else the token `tokenizer.json`'s
//! post-processor puts before a text (Llama 3), and `bos` overrides both.
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
    let mut bos = Bos::Configured;
    #[allow(unused_mut)]
    let mut config = RuntimeConfig::default();
    for option in args.filter_map(|option| option.into_string().ok()) {
        if let Some(value) = option.strip_prefix("chunks=") {
            chunks = value.parse()?;
        } else if let Some(value) = option.strip_prefix("bos=") {
            bos = match value {
                "none" => Bos::Disabled,
                id => Bos::Token(id.parse()?),
            };
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
    let bos = match bos {
        Bos::Configured => configured_bos(&model_path, &tokenizer)?,
        Bos::Token(id) => Some(id),
        Bos::Disabled => None,
    };
    println!("bos={}", bos.map_or_else(|| "none".to_owned(), |id| id.to_string()));
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

/// Where the BOS token comes from.
enum Bos {
    /// The model's tokenizer configuration.
    Configured,
    Token(u32),
    Disabled,
}

/// The BOS token the model's tokenizer adds in front of a text, if any, as
/// Hugging Face's tokenizers decide: an explicit `add_bos_token` in
/// `tokenizer_config.json` (with `bos_token` as a string or an added-token
/// object), otherwise the one token `tokenizer.json`'s post-processor puts
/// before the text. Llama 3 declares its BOS only there.
fn configured_bos(
    model: &std::path::Path,
    tokenizer: &Tokenizer,
) -> Result<Option<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let config = fs::read_to_string(model.join("tokenizer_config.json"))
        .ok()
        .map(|text| serde_json::from_str::<serde_json::Value>(&text))
        .transpose()?;
    let declared = config.as_ref().and_then(|config| config.get("add_bos_token")?.as_bool());
    match declared {
        Some(false) => Ok(None),
        Some(true) => {
            let token = config.as_ref().and_then(|config| {
                let token = config.get("bos_token")?;
                token.as_str().or_else(|| token.get("content")?.as_str())
            });
            let token = token.ok_or("add_bos_token is set without a bos_token")?;
            Ok(Some(
                tokenizer
                    .token_to_id(token)
                    .ok_or_else(|| format!("{token} is not in the vocabulary"))?,
            ))
        },
        None => {
            let probe = "a";
            let plain = tokenizer.encode(probe, false)?;
            let special = tokenizer.encode(probe, true)?;
            let (plain, special) = (plain.get_ids(), special.get_ids());
            Ok(special
                .split_first()
                .filter(|(_, rest)| special.len() > plain.len() && rest.starts_with(plain))
                .map(|(&bos, _)| bos))
        },
    }
}
