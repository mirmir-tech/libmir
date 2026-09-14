use std::{io::Write, sync::Arc};

use models::{layout::ModelLayout, tokenizer::TextTokenizer};
use runtime::{
    backend::{ModelHandle, PrefillRequest, SamplingLogits},
    kv::BlockTable,
    tuning::TuningMode,
};
use serde_json::json;
use uuid::Uuid;

use super::{BenchmarkConfig, greedy_token};
use crate::native::{
    error::Result,
    model::{DecodeInput, LoadedModel},
    prefill::MetalPrefillBatch,
};

#[test]
#[ignore = "real Qwen semantic smoke on linked MLX; set MIRMIR_BENCH_MODEL"]
fn answers_arithmetic_in_scalar_and_dynamic_batch() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let context = std::env::var("MIRMIR_BENCH_SEMANTIC_CONTEXT")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let prompts = (10..15)
        .map(|books| prompt(&tokenizer, books, context))
        .collect::<Result<Vec<_>>>()?;
    let mut settings = super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 0;
    metal.set_max_batch_requests(prompts.len());
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let stop_tokens = tokenizer.stop_token_ids();
    assert!(!stop_tokens.is_empty(), "semantic smoke requires EOS tokens");
    for batch in [1, prompts.len()] {
        let mut answers = Vec::new();
        for chunk in prompts.chunks(batch) {
            for tokens in generate(&mut model, chunk, &stop_tokens)? {
                answers.push(tokenizer.decode(&tokens)?);
            }
        }
        writeln!(
            std::io::stderr().lock(),
            "semantic.sample: {}",
            json!({"mlx": mirtal::version().map_err(crate::engine::Error::from)?,
                "batch": batch, "context": context, "answers": answers, "expected": [7, 8, 9, 10, 11]}),
        )?;
        for (index, answer) in answers.iter().enumerate() {
            assert_eq!(answer.trim(), (index + 7).to_string(), "batch {batch}, row {index}");
        }
    }
    Ok(())
}

pub(super) fn generate(
    model: &mut LoadedModel,
    prompts: &[Vec<u32>],
    stop_tokens: &[u32],
) -> Result<Vec<Vec<u32>>> {
    let mut active = start(model, prompts)?;
    let mut tokens = vec![Vec::new(); prompts.len()];
    for _ in 0..16 {
        let mut pending = Vec::new();
        for (row, input) in active {
            if stop_tokens.contains(&input.token) {
                model.release_session(input.session)?;
            } else {
                tokens[row].push(input.token);
                pending.push((row, input));
            }
        }
        if pending.is_empty() {
            return Ok(tokens);
        }
        let inputs = pending.iter().map(|(_, input)| *input).collect::<Vec<_>>();
        let outputs = if inputs.len() == 1 {
            let input = inputs[0];
            vec![model.decode(input.session, input.token, input.sampling)?]
        } else {
            model.decode_batch(&inputs)?
        };
        assert_eq!(outputs.len(), pending.len());
        for ((_, input), output) in pending.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
        }
        active = pending;
    }
    for (_, input) in active {
        model.release_session(input.session)?;
    }
    Err(super::Error::Benchmark(
        "semantic smoke did not reach EOS within 16 tokens".into(),
    ))
}

pub(super) fn start(
    model: &mut LoadedModel,
    prompts: &[Vec<u32>],
) -> Result<Vec<(usize, DecodeInput)>> {
    if prompts.len() == 1 {
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, &prompts[0], &[], SamplingLogits::None, None, &mut |_| {})?;
        return Ok(vec![(
            0,
            DecodeInput {
                session,
                token: greedy_token(&output.output)?,
                sampling: SamplingLogits::None,
            },
        )]);
    }
    let requests = prompts
        .iter()
        .map(|prompt| {
            (
                PrefillRequest {
                    model: ModelHandle {
                        id: model.info.manifest.id.clone(),
                        backend: "metal".into(),
                    },
                    session_id: Uuid::new_v4(),
                    prompt_tokens: prompt.clone(),
                    cache_checkpoints: Vec::new(),
                    block_table: BlockTable::with_block_size(16),
                    cached_tokens: 0,
                    generation_tokens: None,
                    sampling_logits: SamplingLogits::None,
                },
                SamplingLogits::None,
            )
        })
        .collect();
    let (batch, _) = MetalPrefillBatch::prepare(model, requests, None)?;
    let preserve = std::env::var("MIRMIR_BENCH_PRESERVE_PREFILL_COHORT").as_deref() == Ok("1");
    if preserve {
        batch.preserve_cohort_for_benchmark()?;
    }
    for iteration in 0..512 {
        let step = batch.execute_step(model, 2048)?;
        if preserve && iteration == 0 && prompts.iter().all(|prompt| prompt.len() == 2048) {
            assert_eq!(step.events.len(), 5);
            assert!(step.events.iter().all(|(_, event)| event.count().current() == 409));
        }
        if step.complete {
            return batch
                .finish()?
                .into_iter()
                .enumerate()
                .map(|(row, finished)| {
                    Ok((
                        row,
                        DecodeInput {
                            session: finished.request.session_id,
                            token: greedy_token(&finished.native.output)?,
                            sampling: SamplingLogits::None,
                        },
                    ))
                })
                .collect();
        }
    }
    Err(super::Error::Benchmark("semantic prefill did not finish".into()))
}

pub(super) fn prompt(tokenizer: &TextTokenizer, books: usize, context: usize) -> Result<Vec<u32>> {
    let header = tokenizer.encode_with_special_tokens("<|im_start|>user\n", false)?.token_ids;
    let ending = tokenizer.encode_with_special_tokens(&format!(
        "A shelf has {books} books. Three books are borrowed. How many books remain? Answer only the number.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ), false)?.token_ids;
    let padding = if context == 0 {
        0
    } else {
        context
            .checked_sub(header.len() + ending.len())
            .ok_or_else(|| super::Error::Benchmark("semantic context is too short".into()))?
    };
    let filler = tokenizer
        .encode_with_special_tokens("Background: the library is open to readers. ", false)?
        .token_ids;
    Ok(header
        .into_iter()
        .chain(filler.into_iter().cycle().take(padding))
        .chain(ending)
        .collect())
}
