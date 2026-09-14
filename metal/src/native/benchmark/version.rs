use std::{io::Write, sync::Arc, time::Instant};

use models::{layout::ModelLayout, tokenizer::TextTokenizer};
use runtime::{backend::SamplingLogits, tuning::TuningMode};
use serde_json::json;
use uuid::Uuid;

use super::{BenchmarkConfig, greedy_token};
use crate::native::{
    error::Result,
    model::{DecodeInput, LoadedModel},
};

struct Sample {
    tokens: Vec<Vec<u32>>,
    prefill_ms: Vec<f64>,
    decode_ms: Vec<f64>,
}

#[test]
#[ignore = "frozen-plan cross-MLX version gate; set MIRMIR_BENCH_MODEL"]
fn measures_frozen_version_cohort() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let batch = super::positive_env("MIRMIR_BENCH_BATCH", 1)?;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let stop_tokens = tokenizer.stop_token_ids();
    let prompts = (0..batch)
        .map(|row| prompt(&tokenizer, config.prompt_tokens, row))
        .collect::<Result<Vec<_>>>()?;
    let mut settings = super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 0;
    metal.set_max_batch_requests(batch);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    writeln!(
        std::io::stderr().lock(),
        "version.config: {}",
        json!({
            "mlx": mirtal::version().map_err(crate::engine::Error::from)?,
            "context": config.prompt_tokens, "batch": batch,
            "steps": config.decode_tokens, "warmup": config.warmup,
            "samples": config.samples, "tuning": "disabled", "prefix_cache": false,
            "fusion": model.expert_fusion_summary(), "prompts": prompts,
        })
    )?;
    let mut previous = None;
    for round in 0..config.warmup + config.samples {
        let Sample { tokens, prefill_ms, decode_ms } =
            run(&mut model, &prompts, config.decode_tokens)?;
        assert!(
            tokens.iter().flatten().all(|token| !stop_tokens.contains(token)),
            "version timing crossed EOS; use a prompt with a longer answer"
        );
        if let Some(previous) = &previous {
            assert_eq!(previous, &tokens, "repeated generation changed tokens");
        }
        let memory = crate::engine::memory_stats()?;
        let mut hash = blake3::Hasher::new();
        for token in tokens.iter().flatten() {
            hash.update(&token.to_le_bytes());
        }
        let text = tokens
            .iter()
            .map(|row| tokenizer.decode(row))
            .collect::<models::Result<Vec<_>>>()?;
        writeln!(
            std::io::stderr().lock(),
            "version.sample: {}",
            json!({
                "round": round, "measured": round >= config.warmup,
                "prefill_ms": prefill_ms, "decode_ms": decode_ms,
                "digest": hash.finalize().to_string(), "tokens": tokens, "text": text,
            "memory_after_release": {"active": memory.active, "cached": memory.cached, "peak": memory.peak},
            })
        )?;
        previous = Some(tokens);
    }
    Ok(())
}

fn run(model: &mut LoadedModel, prompts: &[Vec<u32>], steps: usize) -> Result<Sample> {
    let mut inputs = Vec::new();
    let mut tokens = Vec::new();
    let mut prefill_ms = Vec::new();
    for prompt in prompts {
        let session = Uuid::new_v4();
        let started = Instant::now();
        let output =
            model.prefill(session, prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
        let token = greedy_token(&output.output)?;
        model.stream().synchronize()?;
        prefill_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        inputs.push(DecodeInput {
            session,
            token,
            sampling: SamplingLogits::None,
        });
        tokens.push(vec![token]);
    }
    let mut decode_ms = Vec::new();
    for _ in 0..steps {
        let started = Instant::now();
        let outputs = if inputs.len() == 1 {
            let input = inputs[0];
            vec![model.decode(input.session, input.token, input.sampling)?]
        } else {
            model.decode_batch(&inputs)?
        };
        for ((input, row), output) in inputs.iter_mut().zip(&mut tokens).zip(outputs) {
            input.token = greedy_token(&output)?;
            row.push(input.token);
        }
        model.stream().synchronize()?;
        decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    for input in inputs {
        model.release_session(input.session)?;
    }
    Ok(Sample { tokens, prefill_ms, decode_ms })
}

fn prompt(tokenizer: &TextTokenizer, context: usize, row: usize) -> Result<Vec<u32>> {
    let header = tokenizer.encode_with_special_tokens("<|im_start|>user\n", false)?.token_ids;
    let filler = tokenizer.encode_with_special_tokens(
        "This is background material about a library. Books are sorted by title. Readers can borrow books and return them at the front desk. ", false,
    )?.token_ids;
    let ending = tokenizer.encode_with_special_tokens(&format!(
        "\nTask {}: Write a detailed practical guide for managing a library with {} shelves. Cover book organization, borrowing, returns and lost books in four numbered sections. Use at least 300 words.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        row + 1, row + 10,
    ), false)?.token_ids;
    let padding = context.checked_sub(header.len() + ending.len()).ok_or_else(|| {
        super::Error::Benchmark("context is too short for the version-gate prompt".into())
    })?;
    Ok(header
        .into_iter()
        .chain(filler.into_iter().cycle().take(padding))
        .chain(ending)
        .collect())
}
