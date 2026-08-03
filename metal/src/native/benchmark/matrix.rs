use std::{
    env,
    io::Write,
    time::{Duration, Instant},
};

use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{BenchmarkConfig, greedy_token};
use crate::native::{
    error::{Error, Result},
    model::LoadedModel,
};

const DEFAULT_CONTEXTS: &[usize] = &[128, 512, 2_048, 8_192];
const GREEDY_VERIFY_CONTEXT: usize = 128;
const GREEDY_VERIFY_DECODE_TOKENS: usize = 16;
const GEMMA_4_GREEDY_DIGEST: [u8; 32] = [
    0xf1, 0xf6, 0x3f, 0x73, 0x4a, 0xb1, 0xaf, 0x76, 0x48, 0xd2, 0x9b, 0xf5, 0x61, 0x2d, 0xeb, 0x3e,
    0xeb, 0x35, 0xbe, 0x34, 0x8e, 0xc7, 0xef, 0x9a, 0x3c, 0xf8, 0x96, 0x18, 0x6a, 0x56, 0x24, 0xe9,
];
const QWEN_3_6_GREEDY_DIGEST: [u8; 32] = [
    0x71, 0x14, 0xc0, 0xb6, 0x8b, 0x9d, 0x43, 0x01, 0xb6, 0x05, 0xfa, 0x6d, 0x2d, 0x25, 0x8b, 0x4a,
    0x07, 0x55, 0xda, 0xd9, 0x4f, 0x8e, 0xc1, 0xc7, 0x87, 0xd4, 0xc7, 0x9b, 0xb7, 0xfe, 0x0c, 0x1c,
];

#[derive(Debug)]
struct Sample {
    prefill: Duration,
    decode: Duration,
    tokens: Vec<u32>,
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn measures_native_hybrid_moe_context_matrix() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let contexts = contexts()?;
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&config.manifest(), &mut ignored)?;
    let mut report = std::io::stderr().lock();

    for (index, context) in contexts.into_iter().enumerate() {
        let samples = collect_samples(&mut model, &config, context, index)?;
        let prefill = median(samples.iter().map(|sample| sample.prefill).collect());
        let decode = median(samples.iter().map(|sample| sample.decode).collect());
        writeln!(
            report,
            "context_matrix.benchmark: context={context}, samples={}, prefill={:.2} tok/s ({:.2}ms), decode={:.2} tok/s ({:.2}ms), greedy_digest={}",
            config.samples,
            tokens_per_second(u32::try_from(context)?, prefill),
            milliseconds(prefill),
            tokens_per_second(u32::try_from(config.decode_tokens)?, decode),
            milliseconds(decode),
            digest(&samples),
        )?;
        if show_tokens() {
            for (sample_index, sample) in samples.iter().enumerate() {
                writeln!(
                    report,
                    "context_matrix.greedy_tokens: context={context}, sample={sample_index}, tokens={:?}",
                    sample.tokens
                )?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn preserves_hybrid_moe_greedy_sequence_after_128_token_prefill() -> Result<()> {
    let config = BenchmarkConfig {
        model: super::model_path()?,
        decode_tokens: GREEDY_VERIFY_DECODE_TOKENS,
        prompt_tokens: GREEDY_VERIFY_CONTEXT,
        samples: 1,
        warmup: 0,
    };
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&config.manifest(), &mut ignored)?;
    let contract = model
        .info
        .contract
        .as_ref()
        .ok_or_else(|| Error::Benchmark("context matrix requires a decoder model".into()))?;
    let expected = expected_digest(crate::engine::lowering::plan(&contract.semantic)?.runtime())?;
    let samples = collect_samples(&mut model, &config, GREEDY_VERIFY_CONTEXT, 0)?;
    assert_eq!(digest(&samples).as_bytes(), expected);
    Ok(())
}

fn expected_digest(runtime: crate::engine::lowering::DecoderRuntime) -> Result<&'static [u8; 32]> {
    match runtime {
        crate::engine::lowering::DecoderRuntime::DenseAndRouted => Ok(&GEMMA_4_GREEDY_DIGEST),
        crate::engine::lowering::DecoderRuntime::SharedRouted => Ok(&QWEN_3_6_GREEDY_DIGEST),
        crate::engine::lowering::DecoderRuntime::Dense => {
            Err(Error::Benchmark("context matrix digest requires a hybrid MoE model".into()))
        },
        crate::engine::lowering::DecoderRuntime::ClampedRouted => {
            Err(Error::Benchmark("context matrix has no clamped-routed reference digest".into()))
        },
    }
}

fn collect_samples(
    model: &mut LoadedModel,
    config: &BenchmarkConfig,
    context: usize,
    matrix_index: usize,
) -> Result<Vec<Sample>> {
    let total = config.warmup + config.samples;
    let delay = benchmark_delay()?;
    let mut samples = Vec::with_capacity(config.samples);
    for sample_index in 0..total {
        let prompt = prompt(context, matrix_index, sample_index)?;
        let sample = run_sample(model, &prompt, config.decode_tokens)?;
        if sample_index >= config.warmup {
            samples.push(sample);
        }
        if sample_index + 1 < total {
            std::thread::sleep(delay);
        }
    }
    Ok(samples)
}

fn benchmark_delay() -> Result<Duration> {
    let milliseconds = env::var("MIRMIR_BENCH_DELAY_MS")
        .unwrap_or_else(|_| "0".into())
        .parse::<u64>()?;
    Ok(Duration::from_millis(milliseconds))
}

fn run_sample(model: &mut LoadedModel, prompt: &[u32], decode_tokens: usize) -> Result<Sample> {
    let session = Uuid::new_v4();
    let mut ignored = |_event| {};
    let started = Instant::now();
    let output = model.prefill(session, prompt, &[], SamplingLogits::None, None, &mut ignored)?;
    let prefill = started.elapsed();
    let mut token = greedy_token(&output.output)?;
    drop(output);

    let started = Instant::now();
    let mut tokens = Vec::with_capacity(decode_tokens + 1);
    tokens.push(token);
    for _ in 0..decode_tokens {
        let output = model.decode(session, token, SamplingLogits::None)?;
        token = greedy_token(&output)?;
        tokens.push(token);
    }
    let decode = started.elapsed();
    model.release_session(session)?;
    Ok(Sample { prefill, decode, tokens })
}

fn contexts() -> Result<Vec<usize>> {
    let values = env::var("MIRMIR_BENCH_CONTEXTS").unwrap_or_else(|_| {
        DEFAULT_CONTEXTS.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
    });
    values
        .split(',')
        .map(|value| {
            let value = value.trim().parse::<usize>()?;
            if value == 0 {
                return Err(Error::Benchmark("MIRMIR_BENCH_CONTEXTS must be positive".into()));
            }
            Ok(value)
        })
        .collect()
}

fn prompt(context: usize, matrix_index: usize, sample_index: usize) -> Result<Vec<u32>> {
    let seed = matrix_index
        .checked_mul(31)
        .and_then(|value| value.checked_add(sample_index))
        .ok_or_else(|| Error::Benchmark("benchmark prompt seed overflow".into()))?;
    (0..context)
        .map(|index| Ok(u32::try_from((seed * 1_009 + index) % 100_000 + 1_000)?))
        .collect()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn digest(samples: &[Sample]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    for sample in samples {
        for token in &sample.tokens {
            hasher.update(&token.to_le_bytes());
        }
    }
    hasher.finalize()
}

fn tokens_per_second(tokens: u32, elapsed: Duration) -> f64 {
    f64::from(tokens) / elapsed.as_secs_f64()
}

fn milliseconds(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1_000.0
}

fn show_tokens() -> bool {
    matches!(
        env::var("MIRMIR_BENCH_SHOW_TOKENS").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}
