use std::{env, io::Write, time::Instant};

use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{BenchmarkConfig, greedy_token};
use crate::native::{
    error::{Error, Result},
    model::{DecodeInput, LoadedModel},
};

const DEFAULT_CONTEXT: usize = 8_192;
const DEFAULT_BATCH: usize = 5;

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn measures_shared_prefix_decode_batch() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let context = value("MIRMIR_BENCH_CONTEXT", DEFAULT_CONTEXT)?;
    let batch = value("MIRMIR_BENCH_BATCH", DEFAULT_BATCH)?;
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load_with_config(
        &config.manifest(),
        super::diagnostics::isolated_config(),
        &mut ignored,
    )?;
    let prompt = (0..context)
        .map(|index| Ok(u32::try_from(index % 100_000 + 1_000)?))
        .collect::<Result<Vec<_>>>()?;

    let seed = Uuid::new_v4();
    drop(model.prefill(seed, &prompt, &[], SamplingLogits::None, None, &mut ignored)?);
    model.release_session(seed)?;

    let mut inputs = (0..batch)
        .map(|_| {
            let session = Uuid::new_v4();
            let output =
                model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut ignored)?;
            Ok(DecodeInput {
                session,
                token: greedy_token(&output.output)?,
                sampling: SamplingLogits::None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let started = Instant::now();
    for _ in 0..config.decode_tokens {
        let outputs = model.decode_batch(&inputs)?;
        let mut expected = None;
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
            if expected.replace(input.token).is_some_and(|token| token != input.token) {
                return Err(Error::Benchmark(
                    "identical shared-prefix rows produced different greedy tokens".into(),
                ));
            }
        }
    }
    let elapsed = started.elapsed();
    writeln!(
        std::io::stderr().lock(),
        "shared_prefix_batch.benchmark: context={context}, batch={batch}, decode_tokens={}, aggregate={:.2} tok/s ({:.2}ms)",
        config.decode_tokens,
        f64::from(u32::try_from(batch)?) * f64::from(u32::try_from(config.decode_tokens)?)
            / elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1_000.0,
    )?;
    Ok(())
}

fn value(name: &str, default: usize) -> Result<usize> {
    Ok(env::var(name).unwrap_or_else(|_| default.to_string()).parse()?)
}
