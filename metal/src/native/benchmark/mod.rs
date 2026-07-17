mod bielik;
mod llama;
mod matrix;
mod qwen;

use std::{
    env,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

use foundation::model::{BackendTarget, ModelFamily, ModelManifest, Quantization};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{
    error::{Error, Result},
    model::{LoadedModel, NativeOutput},
};

const DEFAULT_DECODE_TOKENS: usize = 128;
const DEFAULT_PROMPT_TOKENS: usize = 96;
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_WARMUP: usize = 2;

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn measures_native_hybrid_moe_decode_pipeline() -> Result<()> {
    init_profile_tracing();
    let config = BenchmarkConfig::from_env()?;
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&config.manifest(), &mut ignored)?;
    let prompt = prompt_tokens(config.prompt_tokens)?;

    for _ in 0..config.warmup {
        let _ = run_decode(&mut model, &prompt, config.decode_tokens)?;
    }
    let mut samples = (0..config.samples)
        .map(|_| run_decode(&mut model, &prompt, config.decode_tokens))
        .collect::<Result<Vec<_>>>()?;
    samples.sort_unstable();

    let decode_tokens = u32::try_from(config.decode_tokens)?;
    let median = samples[samples.len() / 2];
    let best = samples[0];
    let mut report = std::io::stderr().lock();
    writeln!(
        report,
        "decode.benchmark: samples={}, warmup={}, prompt_tokens={}, decode_tokens={}, median={:.2} tok/s ({:.2}ms), best={:.2} tok/s ({:.2}ms)",
        config.samples,
        config.warmup,
        config.prompt_tokens,
        config.decode_tokens,
        tokens_per_second(decode_tokens, median),
        milliseconds(median),
        tokens_per_second(decode_tokens, best),
        milliseconds(best),
    )?;
    Ok(())
}

fn init_profile_tracing() {
    let enabled = ["MIRMIR_METAL_PROFILE_COMPONENTS", "MIRMIR_METAL_PROFILE_GRAPH_BUILD"]
        .into_iter()
        .any(|name| matches!(env::var(name).as_deref(), Ok("1" | "true" | "TRUE" | "yes" | "YES")));
    if enabled {
        drop(
            tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .without_time()
                .try_init(),
        );
    }
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn keeps_kv_state_for_interleaved_sessions() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&config.manifest(), &mut ignored)?;
    let first_session = Uuid::new_v4();
    let second_session = Uuid::new_v4();
    let first = model.prefill(first_session, &[1, 2], SamplingLogits::None, &mut ignored)?;
    let second = model.prefill(second_session, &[3, 4], SamplingLogits::None, &mut ignored)?;
    let first_token = greedy_token(&first.output)?;
    let _second_token = greedy_token(&second.output)?;

    assert_eq!(model.session_cached_tokens(first_session)?, 2);
    assert_eq!(model.session_cached_tokens(second_session)?, 2);
    drop(model.decode(first_session, first_token, SamplingLogits::None)?);
    assert_eq!(model.session_cached_tokens(first_session)?, 3);
    assert_eq!(model.session_cached_tokens(second_session)?, 2);
    Ok(())
}

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn reuses_exact_device_prefix_snapshot() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load(&config.manifest(), &mut ignored)?;
    let prompt = [1, 2];
    let first = model.prefill(Uuid::new_v4(), &prompt, SamplingLogits::None, &mut ignored)?;
    let second = model.prefill(Uuid::new_v4(), &prompt, SamplingLogits::None, &mut ignored)?;

    assert_eq!(first.prefix_cache_tokens, 0);
    assert_eq!(second.prefix_cache_tokens, prompt.len());
    assert_eq!(greedy_token(&first.output)?, greedy_token(&second.output)?);

    let extended_session = Uuid::new_v4();
    let extended =
        model.prefill(extended_session, &[1, 2, 3], SamplingLogits::None, &mut ignored)?;
    let extended_token = greedy_token(&extended.output)?;
    assert_eq!(extended.prefix_cache_tokens, prompt.len());
    assert_eq!(model.session_cached_tokens(extended_session)?, 3);
    drop(model.decode(extended_session, extended_token, SamplingLogits::None)?);
    assert_eq!(model.session_cached_tokens(extended_session)?, 4);
    Ok(())
}

fn run_decode(model: &mut LoadedModel, prompt: &[u32], decode_tokens: usize) -> Result<Duration> {
    let session = Uuid::new_v4();
    let mut ignored = |_event| {};
    let output = model.prefill(session, prompt, SamplingLogits::None, &mut ignored)?;
    let mut token = greedy_token(&output.output)?;
    let started = Instant::now();
    for _ in 0..decode_tokens {
        let output = model.decode(session, token, SamplingLogits::None)?;
        token = greedy_token(&output)?;
    }
    Ok(started.elapsed())
}

fn greedy_token(output: &NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(*token),
        NativeOutput::Logits(_) => Err(Error::Benchmark("greedy pipeline returned logits".into())),
    }
}

fn prompt_tokens(count: usize) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(count);
    for token in 1..=count {
        tokens.push(u32::try_from(token)?);
    }
    Ok(tokens)
}

fn tokens_per_second(tokens: u32, elapsed: Duration) -> f64 {
    f64::from(tokens) / elapsed.as_secs_f64()
}

fn milliseconds(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1_000.0
}

#[derive(Debug)]
struct BenchmarkConfig {
    model: PathBuf,
    decode_tokens: usize,
    prompt_tokens: usize,
    samples: usize,
    warmup: usize,
}

impl BenchmarkConfig {
    fn from_env() -> Result<Self> {
        Ok(Self {
            model: model_path()?,
            decode_tokens: positive_env("MIRMIR_BENCH_DECODE_TOKENS", DEFAULT_DECODE_TOKENS)?,
            prompt_tokens: positive_env("MIRMIR_BENCH_PROMPT_TOKENS", DEFAULT_PROMPT_TOKENS)?,
            samples: positive_env("MIRMIR_BENCH_SAMPLES", DEFAULT_SAMPLES)?,
            warmup: env_usize("MIRMIR_BENCH_WARMUP", DEFAULT_WARMUP)?,
        })
    }

    fn manifest(&self) -> ModelManifest {
        ModelManifest {
            id: "native-decode-benchmark".into(),
            family: ModelFamily::Unknown,
            path: self.model.to_string_lossy().into_owned(),
            tokenizer_path: None,
            context_len: self.prompt_tokens + self.decode_tokens,
            quantization: Quantization::Int4,
            preferred_backends: vec![BackendTarget::Metal],
        }
    }
}

fn model_path() -> Result<PathBuf> {
    env::var_os("MIRMIR_BENCH_MODEL")
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or_else(|| Error::Benchmark("set MIRMIR_BENCH_MODEL or MODEL".into()))
}

fn positive_env(name: &str, default: usize) -> Result<usize> {
    let value = env_usize(name, default)?;
    if value == 0 {
        return Err(Error::Benchmark(format!("{name} must be greater than zero")));
    }
    Ok(value)
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    Ok(value.parse()?)
}
