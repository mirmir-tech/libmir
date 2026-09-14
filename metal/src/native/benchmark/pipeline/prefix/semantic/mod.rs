mod execution;
mod prompts;
mod report;

use std::{io::Write, sync::Arc};

use models::{layout::ModelLayout, tokenizer::TextTokenizer};
use runtime::{backend::SamplingLogits, tuning::TuningMode};

use super::{BenchmarkConfig, prefill};
use crate::native::{
    error::Result,
    model::{DecodeInput, LoadedModel},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Finish {
    Eos,
    Limit,
    Cancelled,
}

struct Answer {
    case: usize,
    tokens: Vec<u32>,
    finish: Finish,
}
struct Row {
    case: usize,
    input: DecodeInput,
    tokens: Vec<u32>,
}

#[test]
#[ignore = "real Qwen semantic retrieval with shared prefixes, mixed sampling, cancellation and refill"]
fn answers_registry_through_mixed_prefix_decode() -> Result<()> {
    let context = super::super::super::positive_env("MIRMIR_BENCH_PREFIX_QUALITY_CONTEXT", 2049)?;
    assert!(context >= 512, "registry context must leave room for its three sections");
    let config = BenchmarkConfig {
        model: super::super::super::model_path()?,
        prompt_tokens: context,
        decode_tokens: prompts::LIMIT,
        samples: 1,
        warmup: 0,
    };
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let (prefix, prompts) = prompts::build(&tokenizer, context)?;
    let stops = tokenizer.stop_token_ids();
    assert!(!stops.is_empty());
    let mut settings = super::super::super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 8;
    metal.set_max_batch_requests(4);
    let blocks = (prompts.iter().map(Vec::len).max().unwrap_or(context) + prompts::LIMIT + 2)
        .div_ceil(metal.kv_cache.block_size)
        * 4;
    metal.kv_cache.block_count = metal.kv_cache.block_count.max(u32::try_from(blocks)?);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let mut references = Vec::new();
    for (case, prompt) in prompts.iter().enumerate() {
        model.clear_prefix_cache();
        references.push(execution::scalar(&mut model, case, prompt, &stops)?);
    }
    model.clear_prefix_cache();
    let (seed, hit) = prefill(&mut model, &prefix, SamplingLogits::Full)?;
    assert_eq!(hit, 0);
    model.release_session(seed.session)?;
    let (answers, packed, full, hits) = execution::mixed(&mut model, &prompts, &stops, context)?;
    assert!(model.sessions.is_empty());
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    writeln!(
        std::io::stderr().lock(),
        "prefix.quality.coverage: {}",
        serde_json::json!({
            "context":context,"prompt_lengths":prompts.iter().map(Vec::len).collect::<Vec<_>>(),
            "packed_outputs":packed,"full_logits_requests":full,"prefix_hits":hits,"resident_arenas":0
        })
    )?;
    report::check(&references, &answers, &tokenizer)
}
