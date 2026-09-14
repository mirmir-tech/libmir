use std::{io::Write, sync::Arc};

use models::{layout::ModelLayout, tokenizer::TextTokenizer};
use runtime::{backend::SamplingLogits, tuning::TuningMode};
use uuid::Uuid;

use super::super::{BenchmarkConfig, diagnostics, greedy_token, semantic};
use crate::native::{error::Result, model::LoadedModel, prefix::PrefixCache};

#[test]
#[ignore = "real Qwen recurrent-prefix budget regression; set MIRMIR_BENCH_MODEL"]
fn enforces_qwen_prefix_budget_including_recurrence() -> Result<()> {
    const MIB: usize = 1024 * 1024;
    let mut config = BenchmarkConfig::from_env()?;
    config.prompt_tokens = 2048;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let prompt = semantic::prompt(&tokenizer, 10, 2048)?;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 4;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    for budget in [192 * MIB, 96 * MIB] {
        model.prefixes = PrefixCache::new(4, budget);
        for repeat in 0..2 {
            let session = Uuid::new_v4();
            let result = model.prefill(
                session,
                &prompt,
                &[],
                SamplingLogits::None,
                Some(16),
                &mut |_| {},
            )?;
            let token = greedy_token(&result.output)?;
            assert_eq!(tokenizer.decode(&[token])?.trim(), "7");
            let next = greedy_token(&model.decode(session, token, SamplingLogits::None)?)?;
            assert!(tokenizer.stop_token_ids().contains(&next), "expected EOS");
            let hit = budget == 192 * MIB && repeat == 1;
            assert_eq!(
                result.prefix_cache_tokens,
                if hit {
                    prompt.len()
                } else {
                    0
                }
            );
            let bytes = model.prefixes.resident_bytes();
            assert!(bytes <= budget);
            if budget == 192 * MIB {
                assert_eq!(model.prefixes.group_count(), 1);
                assert!(model.prefixes.recurrent_bytes() >= 60 * MIB);
                assert!(bytes > 96 * MIB, "old K/V-only estimate would fit the smaller budget");
            } else {
                assert_eq!(model.prefixes.group_count(), 0);
            }
            writeln!(
                std::io::stderr().lock(),
                "prefix.memory: {}",
                serde_json::json!({
                    "budget_bytes": budget, "repeat": repeat, "hit": hit,
                    "accounted_bytes": bytes, "recurrent_bytes": model.prefixes.recurrent_bytes(),
                    "answer": tokenizer.decode(&[token])?,
                })
            )?;
            model.release_session(session)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "real Qwen continuation from compacted recurrent checkpoint; set MIRMIR_BENCH_MODEL"]
fn restores_qwen_from_retained_checkpoint() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.diagnostics.prefix_retention = crate::config::PrefixRetention::CompactCheckpoint;
    metal.cache.prefix_cache_entries = 4;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let first = semantic::prompt(&tokenizer, 10, 2048)?;
    let branch = semantic::prompt(&tokenizer, 11, 2048)?;
    assert_eq!(first[..1024], branch[..1024]);
    for (round, prompt, expected, cached) in [
        (0, &first, "7", 0),
        (1, &branch, "8", 1024),
        (2, &branch, "8", 2048),
        (3, &branch, "8", 0),
    ] {
        if round == 3 {
            model.prefixes = PrefixCache::new(0, 0);
        }
        let session = Uuid::new_v4();
        let result =
            model.prefill(session, prompt, &[1024], SamplingLogits::None, Some(16), &mut |_| {})?;
        assert_eq!(result.prefix_cache_tokens, cached);
        let token = greedy_token(&result.output)?;
        assert_eq!(tokenizer.decode(&[token])?.trim(), expected);
        let next = greedy_token(&model.decode(session, token, SamplingLogits::None)?)?;
        assert!(tokenizer.stop_token_ids().contains(&next), "expected EOS");
        writeln!(
            std::io::stderr().lock(),
            "checkpoint.semantic: {}",
            serde_json::json!({
                "round": round, "cached_tokens": cached, "answer": expected,
                "convolution_bytes": model.prefixes.convolution_bytes()?,
                "prefix_bytes": model.prefixes.resident_bytes(),
            })
        )?;
        model.release_session(session)?;
    }
    Ok(())
}
