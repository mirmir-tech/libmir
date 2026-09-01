use std::{env, path::PathBuf};

use cuda::{CudaKernelAdmission, CudaNumericalPolicy, CudaOutputHeadPolicy};
use libmir::{Conversation, GenerationOverrides, Library, Message, RuntimeConfig, SamplingLogits};
use runtime::backend::LogitsTrace;

mod metrics;

use metrics::Metrics;

const PROMPTS: [&str; 3] = [
    "Explain continuous batching in an LLM inference server.",
    "Write a short proof that the square root of two is irrational.",
    "Wymień trzy przyczyny powstawania zorzy polarnej.",
];

struct Trace {
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model = env::args_os().nth(1).map(PathBuf::from).ok_or("missing model path")?;
    let steps = env::args().nth(2).map_or(Ok(32), |value| value.parse())?;
    let (candidate, label) = candidate()?;
    let traces = trace(&model, steps, CudaOutputHeadPolicy::Bf16)?;
    let metrics = compare(&model, &traces, candidate)?;
    println!("model={}", model.display());
    println!("candidate={label} {}", metrics.report());
    Ok(())
}

fn candidate() -> Result<(CudaOutputHeadPolicy, &'static str), &'static str> {
    match env::var("MIRMIR_OUTPUT_QUALITY_CANDIDATE")
        .as_deref()
        .unwrap_or("fp8-block-refined")
    {
        "bf16" => Ok((CudaOutputHeadPolicy::Bf16, "bf16")),
        "fp8-block-refined" => Ok((CudaOutputHeadPolicy::Fp8BlockRefined, "fp8-block-refined")),
        _ => Err("unsupported output-quality candidate"),
    }
}

fn trace(
    path: &PathBuf,
    steps: usize,
    policy: CudaOutputHeadPolicy,
) -> Result<Vec<Trace>, Box<dyn std::error::Error>> {
    let library = Library::new(runtime(policy));
    let model = library.load(path, GenerationOverrides::default(), &mut |_| {})?;
    let mut traces = Vec::with_capacity(PROMPTS.len());
    for prompt in PROMPTS {
        let tokens = prepared(&model, prompt)?;
        let mut session = model.session();
        let output = session.prefill(&tokens, SamplingLogits::Full, &mut |_| {})?;
        let mut logits = required_logits(output.logits)?;
        let mut generated = Vec::with_capacity(steps);
        let mut recorded = Vec::with_capacity(steps);
        for step in 0..steps {
            let token = greedy(&logits)?;
            generated.push(token);
            let next = if step + 1 < steps {
                Some(required_logits(session.decode(token, SamplingLogits::Full)?.logits)?)
            } else {
                None
            };
            recorded.push(logits);
            let Some(next) = next else {
                break;
            };
            logits = next;
        }
        traces.push(Trace {
            prompt: tokens,
            tokens: generated,
            logits: recorded,
        });
    }
    model.unload()?;
    Ok(traces)
}

fn compare(
    path: &PathBuf,
    traces: &[Trace],
    policy: CudaOutputHeadPolicy,
) -> Result<Metrics, Box<dyn std::error::Error>> {
    let library = Library::new(runtime(policy));
    let model = library.load(path, GenerationOverrides::default(), &mut |_| {})?;
    let mut metrics = Metrics::new();
    for trace in traces {
        let mut session = model.session();
        let output = session.prefill(&trace.prompt, SamplingLogits::Full, &mut |_| {})?;
        let mut actual = required_logits(output.logits)?;
        for (step, (token, expected)) in trace.tokens.iter().zip(&trace.logits).enumerate() {
            metrics.observe(expected, &actual);
            if step + 1 < trace.tokens.len() {
                actual = required_logits(session.decode(*token, SamplingLogits::Full)?.logits)?;
            }
        }
    }
    model.unload()?;
    Ok(metrics)
}

fn runtime(policy: CudaOutputHeadPolicy) -> RuntimeConfig {
    let mut runtime = RuntimeConfig::default();
    runtime.scheduler.max_batch_requests = 1;
    runtime.scheduler.decode_batch_wait_us = 0;
    runtime.cuda.planning.output_head = policy;
    runtime.cuda.planning.numerical = CudaNumericalPolicy::Throughput;
    runtime.cuda.planning.admission = CudaKernelAdmission::Experimental;
    runtime
}

fn prepared(model: &libmir::Model, content: &str) -> libmir::Result<Vec<u32>> {
    Ok(model
        .prepare(&Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: content.into(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: libmir::ToolChoice::default(),
        })?
        .tokens
        .token_ids)
}

fn required_logits(logits: Option<LogitsTrace>) -> Result<Vec<f32>, &'static str> {
    logits.map(|trace| trace.values).ok_or("backend did not return full logits")
}

fn greedy(logits: &[f32]) -> Result<u32, Box<dyn std::error::Error>> {
    let (index, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or("empty logits")?;
    Ok(u32::try_from(index)?)
}
