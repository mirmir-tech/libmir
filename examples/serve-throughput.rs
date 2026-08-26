#![allow(clippy::print_stdout)]

use std::{
    env,
    path::PathBuf,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use libmir::cuda::CudaMoeBatchPolicy;
use libmir::{
    ChatCompletionRequest, ChatMessage, Error, GenerationOverrides, Library, Model, RuntimeConfig,
    SamplingLogits, runtime::RuntimeError,
};

struct Config {
    model: PathBuf,
    sessions: usize,
    prompt_tokens: usize,
    decode_steps: usize,
    #[cfg(feature = "cuda")]
    moe_batch: CudaMoeBatchPolicy,
}

struct WorkerStats {
    prefill_tokens: usize,
    prefill: Duration,
    decode: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let config = Config::parse()?;
    let mut runtime = RuntimeConfig::default();
    runtime.scheduler.max_batch_requests = config.sessions;
    runtime.scheduler.max_batch_tokens = runtime.scheduler.max_batch_tokens.max(config.sessions);
    #[cfg(feature = "metal")]
    {
        runtime.metal.diagnostics.profile_components = enabled("MIRMIR_METAL_PROFILE_COMPONENTS");
        runtime.metal.tuning.cache_directory =
            env::var_os("MIRMIR_METAL_TUNING_CACHE").map(PathBuf::from);
    }
    #[cfg(feature = "cuda")]
    {
        runtime.cuda.planning.moe_batch = config.moe_batch;
    }
    let model =
        Library::new(runtime).load(&config.model, GenerationOverrides::default(), &mut |_| {})?;
    let base = model.prepare(&request(&model))?.tokens.token_ids;
    let prompt = base.iter().copied().cycle().take(config.prompt_tokens).collect::<Vec<_>>();
    let workers = run_workers(&model, &prompt, &config)?;
    report(&config, workers);
    Ok(())
}

fn run_workers(
    model: &Model,
    prompt: &[u32],
    config: &Config,
) -> libmir::Result<(Duration, Duration, Vec<WorkerStats>)> {
    let start = Arc::new(Barrier::new(config.sessions + 1));
    let prefill_done = Arc::new(Barrier::new(config.sessions + 1));
    let decode_step = Arc::new(Barrier::new(config.sessions));
    let decode_done = Arc::new(Barrier::new(config.sessions + 1));
    let mut handles = Vec::with_capacity(config.sessions);
    for _ in 0..config.sessions {
        handles.push(spawn_worker(
            model.clone(),
            prompt.to_vec(),
            config.decode_steps,
            [&start, &prefill_done, &decode_step, &decode_done].map(Arc::clone),
        ));
    }

    let prefill_started = Instant::now();
    start.wait();
    prefill_done.wait();
    let prefill = prefill_started.elapsed();
    let decode_started = Instant::now();
    decode_done.wait();
    let decode = decode_started.elapsed();
    let workers = handles.into_iter().map(join).collect::<libmir::Result<Vec<_>>>()?;
    Ok((prefill, decode, workers))
}

fn spawn_worker(
    model: Model,
    prompt: Vec<u32>,
    decode_steps: usize,
    barriers: [Arc<Barrier>; 4],
) -> thread::JoinHandle<libmir::Result<WorkerStats>> {
    thread::spawn(move || {
        let [start, prefill_done, decode_step, decode_done] = barriers;
        let mut session = model.session();
        start.wait();
        let prefill_started = Instant::now();
        let output = session.prefill(&prompt, SamplingLogits::None, &mut |_| {})?;
        let prefill = prefill_started.elapsed();
        let mut next = required_token(output.next_token)?;
        prefill_done.wait();
        let decode_started = Instant::now();
        for _ in 0..decode_steps {
            decode_step.wait();
            next = required_token(session.decode(next, SamplingLogits::None)?.event.token_id)?;
        }
        let decode = decode_started.elapsed();
        decode_done.wait();
        Ok(WorkerStats {
            prefill_tokens: output.accepted_tokens,
            prefill,
            decode,
        })
    })
}

fn join(handle: thread::JoinHandle<libmir::Result<WorkerStats>>) -> libmir::Result<WorkerStats> {
    let Ok(output) = handle.join() else {
        return Err(RuntimeError::Scheduler("throughput worker panicked".into()).into());
    };
    output
}

fn report(config: &Config, result: (Duration, Duration, Vec<WorkerStats>)) {
    let (prefill_wall, decode_wall, workers) = result;
    let prefill_tokens = workers.iter().map(|stats| stats.prefill_tokens).sum::<usize>();
    let decode_tokens = config.sessions.saturating_mul(config.decode_steps);
    let mean_prefill = mean_ms(workers.iter().map(|stats| stats.prefill));
    let mean_decode = mean_ms(workers.iter().map(|stats| stats.decode));
    println!(
        "sessions prompt_tokens decode_steps prefill_tok/s decode_tok/s mean_prefill_ms mean_decode_ms"
    );
    println!(
        "{} {} {} {:.3} {:.3} {:.3} {:.3}",
        config.sessions,
        config.prompt_tokens,
        config.decode_steps,
        rate(prefill_tokens, prefill_wall),
        rate(decode_tokens, decode_wall),
        mean_prefill,
        mean_decode,
    );
}

fn mean_ms(durations: impl Iterator<Item = Duration>) -> f64 {
    let (total, count) = durations.fold((0.0, 0_u32), |(total, count), duration| {
        (total + duration.as_secs_f64(), count.saturating_add(1))
    });
    total * 1_000.0 / f64::from(count)
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    f64::from(u32::try_from(tokens).unwrap_or(u32::MAX)) / duration.as_secs_f64()
}

fn required_token(token: Option<u32>) -> libmir::Result<u32> {
    token.ok_or_else(|| RuntimeError::Backend("device sampling returned no token".into()).into())
}

fn request(model: &Model) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.handle().id.clone(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Explain continuous batching in an LLM inference server.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(1),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(7),
    }
}

impl Config {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let model = env::args_os()
            .nth(1)
            .or_else(|| env::var_os("MODEL"))
            .map(PathBuf::from)
            .ok_or(Error::MissingEnvironment("MODEL or the first argument"))?;
        let sessions = argument(2, 4)?;
        let prompt_tokens = argument(3, 256)?;
        let decode_steps = argument(4, 32)?;
        if sessions == 0 || prompt_tokens == 0 || decode_steps == 0 {
            return Err("sessions, prompt tokens, and decode steps must be positive".into());
        }
        Ok(Self {
            model,
            sessions,
            prompt_tokens,
            decode_steps,
            #[cfg(feature = "cuda")]
            moe_batch: moe_batch_argument(5)?,
        })
    }
}

#[cfg(feature = "cuda")]
fn moe_batch_argument(index: usize) -> Result<CudaMoeBatchPolicy, Box<dyn std::error::Error>> {
    match env::args().nth(index).as_deref().unwrap_or("auto") {
        "auto" => Ok(CudaMoeBatchPolicy::Auto),
        "w4a4" => Ok(CudaMoeBatchPolicy::W4A4),
        "w4a4-direct" => Ok(CudaMoeBatchPolicy::W4A4Direct),
        "w4a4-hybrid" => Ok(CudaMoeBatchPolicy::W4A4Hybrid),
        "w4a4-bucketed" => Ok(CudaMoeBatchPolicy::W4A4Bucketed),
        "w4a16" => Ok(CudaMoeBatchPolicy::W4A16),
        _ => {
            Err("MoE policy: auto, w4a4, w4a4-direct, w4a4-hybrid, w4a4-bucketed, or w4a16".into())
        },
    }
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    env::args().nth(index).map_or(Ok(default), |value| Ok(value.parse()?))
}

#[cfg(feature = "metal")]
fn enabled(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true" | "TRUE" | "yes" | "YES"))
}
