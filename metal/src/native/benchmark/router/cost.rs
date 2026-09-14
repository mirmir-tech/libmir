use std::{io::Write, sync::Arc, time::Instant};

use super::super::{BenchmarkConfig, diagnostics, greedy_token, semantic};
use crate::{
    config::RouterPrecision,
    native::{
        error::{Error, Result},
        model::LoadedModel,
    },
};

#[test]
#[ignore = "bounded router precision C5/512 cost experiment, abort above 15% spread"]
fn measures_router_precision_cost() -> Result<()> {
    let mut config = BenchmarkConfig::from_env()?;
    config.prompt_tokens = 512;
    config.decode_tokens = 32;
    let tokenizer = models::tokenizer::TextTokenizer::from_layout(
        &models::layout::ModelLayout::inspect(&config.model)?,
    )?;
    let prompts = (10..15)
        .map(|books| semantic::prompt(&tokenizer, books, 512))
        .collect::<Result<Vec<_>>>()?;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.cache.prefix_cache_entries = 0;
    metal.tuning.mode = runtime::tuning::TuningMode::Disabled;
    metal.set_max_batch_requests(5);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 128;
    let mut samples = Vec::new();
    let mut reference = [None, None];
    let warmup = [0, 1, 1, 0, 0, 1];
    let measured = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0];
    for (run, index) in warmup.into_iter().chain(measured).enumerate() {
        let precision = [RouterPrecision::Model, RouterPrecision::Float32][index];
        model.stream.synchronize()?;
        model.stream.set_router_precision_for_test(precision);
        let started = Instant::now();
        let mut inputs = semantic::start(&mut model, &prompts)?
            .into_iter()
            .map(|(_, input)| input)
            .collect::<Vec<_>>();
        model.flush_decode_graphs()?;
        let prefill = started.elapsed().as_secs_f64() * 1000.0;
        let mut tokens = inputs.iter().map(|input| input.token).collect::<Vec<_>>();
        let started = Instant::now();
        for _ in 0..32 {
            let outputs = model.decode_batch(&inputs)?;
            for (input, output) in inputs.iter_mut().zip(outputs) {
                input.token = greedy_token(&output)?;
                tokens.push(input.token);
            }
        }
        model.flush_decode_graphs()?;
        let decode = started.elapsed().as_secs_f64() * 1000.0;
        let memory = crate::engine::memory_stats()?;
        for input in inputs {
            model.release_session(input.session)?;
        }
        assert!(model.sessions.is_empty());
        if let Some(expected) = &reference[index] {
            assert_eq!(&tokens, expected, "within-precision trajectory changed");
        } else {
            reference[index] = Some(tokens.clone());
        }
        writeln!(
            std::io::stderr().lock(),
            "router.cost: {}",
            serde_json::json!({
            "run":run,"measured":run>=6,"precision":format!("{precision:?}"),
            "prefill_ms":prefill,"decode_ms":decode,"active_bytes":memory.active,"cache_bytes":memory.cached,
            "tokens":tokens })
        )?;
        if run >= 6 {
            samples.push((index, prefill, decode));
            for mode in 0..2 {
                for phase in 0..2 {
                    let values = samples
                        .iter()
                        .filter(|(index, _, _)| *index == mode)
                        .map(|(_, pp, tg)| {
                            if phase == 0 {
                                *pp
                            } else {
                                *tg
                            }
                        })
                        .collect::<Vec<_>>();
                    if values.len() >= 2 {
                        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                        let max = values.iter().copied().fold(0.0, f64::max);
                        if max / min > 1.15 {
                            return Err(Error::Benchmark(format!(
                                "router timing unstable: mode {mode}, phase {phase}, spread {}%",
                                (max / min - 1.0) * 100.0
                            )));
                        }
                    }
                }
            }
        }
    }
    model.stream.set_router_precision_for_test(RouterPrecision::Model);
    Ok(())
}
