mod execution;
mod prompts;

use std::{io::Write, sync::Arc};

use execution::{Observation, generate};
use models::{layout::ModelLayout, tokenizer::TextTokenizer};
use prompts::Suite;
use serde_json::json;

use crate::native::{error::Result, model::LoadedModel};

#[test]
#[ignore = "actual Qwen C3/C5 natural-prompt route reuse; set MIRMIR_BENCH_MODEL"]
fn measures_cross_session_expert_reuse() -> Result<()> {
    let mut config = super::super::BenchmarkConfig::from_env()?;
    config.prompt_tokens = 4096;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let mut settings = super::super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = runtime::tuning::TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 0;
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let stop = tokenizer.stop_token_ids();
    assert!(!stop.is_empty());
    for suite in [Suite::Diverse, Suite::Related] {
        let prompts = suite.encode(&tokenizer)?;
        writeln!(
            std::io::stderr().lock(),
            "sharing.prompts: {}",
            json!({
                "suite": suite, "questions": suite.questions(), "prompt_ids": prompts,
            })
        )?;
        for width in [3, 5] {
            let control = generate(&mut model, &prompts[..width], &stop, Observation::Disabled)?;
            let observed = generate(&mut model, &prompts[..width], &stop, Observation::Routes)?;
            assert_eq!(control.tokens, observed.tokens, "route observer changed tokens");
            assert_eq!(control.complete, observed.complete);
            let answers = observed
                .tokens
                .iter()
                .map(|tokens| tokenizer.decode(tokens))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            writeln!(
                std::io::stderr().lock(),
                "sharing.control: {}",
                json!({
                    "suite": suite, "width": width, "matches_reference": true,
                    "tokens": observed.tokens, "complete": observed.complete, "answers": answers,
                })
            )?;
            // All generation finishes before reading the captured device indices.
            for (step, rows, routes) in observed.captures {
                assert_eq!(
                    routes.iter().map(|r| r.layer).collect::<Vec<_>>(),
                    (0..40).collect::<Vec<_>>()
                );
                for route in routes {
                    let shape = route.indices.shape()?;
                    let indices = route.indices.to_vec_u32(&model.stream)?;
                    assert_eq!(indices.len(), rows.len() * 8);
                    assert!(indices.iter().all(|&id| id < 256));
                    writeln!(
                        std::io::stderr().lock(),
                        "sharing.routes: {}",
                        json!({
                            "suite": suite, "width": width, "step": step,
                            "layer": route.layer, "rows": rows, "shape": shape, "indices": indices,
                        })
                    )?;
                }
            }
            assert!(model.sessions.is_empty());
        }
    }
    Ok(())
}
