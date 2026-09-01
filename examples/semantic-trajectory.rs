#![allow(clippy::print_stdout)]

use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    time::Instant,
};

use cuda::{CudaKernelAdmission, CudaNumericalPolicy, CudaOutputHeadPolicy};
use libmir::{Conversation, Error, GenerationOverrides, Library, Message, RuntimeConfig};
use models::generation::{GenerationChannel, OutputNormalizer};
use runtime::backend::SamplingLogits;
use serde::{Deserialize, Serialize};

const HARMONY_ANALYSIS: &str = "<|channel|>analysis<|message|>";

#[derive(Deserialize)]
struct Case {
    key: String,
    prompt: String,
    max_tokens: usize,
}

#[derive(Serialize)]
struct ResultRecord {
    key: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    finish_reason: &'static str,
    text: String,
    reasoning: String,
    token_ids: Vec<u32>,
    elapsed_seconds: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let model_path = argument(1, "model path")?;
    let cases_path = argument(2, "cases JSONL path")?;
    let model =
        Library::new(runtime()?).load(model_path, GenerationOverrides::default(), &mut |_| {})?;
    let clear_prefix_cache = env::var_os("MIRMIR_SEMANTIC_CLEAR_PREFIX_CACHE").is_some();
    let retain_sessions = env::var_os("MIRMIR_SEMANTIC_RETAIN_SESSIONS").is_some();
    let mut retained = Vec::new();
    for line in BufReader::new(File::open(cases_path)?).lines() {
        let case: Case = serde_json::from_str(&line?)?;
        if clear_prefix_cache {
            model.engine().clear_prefix_cache(model.handle())?;
        }
        let mut session = model.session();
        println!("{}", serde_json::to_string(&run_case(&model, &mut session, case)?)?);
        if retain_sessions {
            retained.push(session);
        }
    }
    drop(retained);
    model.unload()?;
    Ok(())
}

fn run_case(
    model: &libmir::Model,
    session: &mut libmir::Session,
    case: Case,
) -> Result<ResultRecord, Box<dyn std::error::Error>> {
    let prepared = model.prepare(&conversation(&case.prompt))?;
    let tokenizer = model.descriptor().tokenizer();
    let suffix = tokenizer.encode_with_special_tokens(HARMONY_ANALYSIS, false)?.token_ids;
    let has_preseed = prepared.tokens.token_ids.ends_with(&suffix);
    let prompt_tokens = prepared.tokens.token_ids.len() - usize::from(has_preseed) * suffix.len();
    let prompt = if has_preseed {
        prepared
            .prompt
            .text
            .strip_suffix(HARMONY_ANALYSIS)
            .ok_or("prompt text and tokenized Harmony preseed disagree")?
    } else {
        &prepared.prompt.text
    };
    let sampling = if env::var_os("MIRMIR_SEMANTIC_FULL_LOGITS").is_some() {
        SamplingLogits::Full
    } else {
        SamplingLogits::None
    };
    let started = Instant::now();
    let output =
        session.prefill(&prepared.tokens.token_ids[..prompt_tokens], sampling, &mut |_| {})?;
    let mut next = selected(output.next_token, output.logits.as_ref())?;
    let mut decoder = tokenizer.decoder();
    let mut normalizer = OutputNormalizer::new(tokenizer, prompt);
    let stops = tokenizer.stop_token_ids();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut generated = 0;
    let mut token_ids = Vec::with_capacity(case.max_tokens);
    let mut finish_reason = "max_tokens";
    while generated < case.max_tokens {
        generated += 1;
        token_ids.push(next);
        let piece = decoder.step(next)?.unwrap_or_default();
        if let Some(delta) = normalizer.push(next, piece) {
            match delta.channel {
                GenerationChannel::Content => text.push_str(&delta.text),
                GenerationChannel::Reasoning => reasoning.push_str(&delta.text),
                GenerationChannel::ToolCalls => {},
            }
        }
        if stops.contains(&next) {
            finish_reason = "stop";
            break;
        }
        let output = session.decode(next, sampling)?;
        next = selected(output.event.token_id, output.logits.as_ref())?;
    }
    Ok(ResultRecord {
        key: case.key,
        prompt_tokens,
        generated_tokens: generated,
        finish_reason,
        text,
        reasoning,
        token_ids,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn selected(
    token: Option<u32>,
    logits: Option<&runtime::backend::LogitsTrace>,
) -> Result<u32, &'static str> {
    if let Some(token) = token {
        return Ok(token);
    }
    logits
        .and_then(|trace| {
            trace.values.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1))
        })
        .and_then(|(index, _)| u32::try_from(index).ok())
        .ok_or("generation returned neither token nor logits")
}

fn conversation(prompt: &str) -> Conversation {
    Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: prompt.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: libmir::ToolChoice::default(),
    }
}

fn runtime() -> Result<RuntimeConfig, &'static str> {
    let mut config = RuntimeConfig::default();
    config.scheduler.max_batch_requests = 1;
    config.scheduler.decode_batch_wait_us = 0;
    match env::var("MIRMIR_SEMANTIC_OUTPUT_HEAD").as_deref().unwrap_or("auto") {
        "auto" => {
            config.cuda.planning.numerical = CudaNumericalPolicy::Throughput;
            config.cuda.planning.admission = CudaKernelAdmission::Experimental;
            config.cuda.planning.output_head = CudaOutputHeadPolicy::Auto;
        },
        "bf16-experimental" => {
            config.cuda.planning.numerical = CudaNumericalPolicy::Throughput;
            config.cuda.planning.admission = CudaKernelAdmission::Experimental;
            config.cuda.planning.output_head = CudaOutputHeadPolicy::Bf16;
        },
        "bf16-stable" => config.cuda.planning.output_head = CudaOutputHeadPolicy::Bf16,
        _ => {
            return Err(
                "MIRMIR_SEMANTIC_OUTPUT_HEAD must be auto, bf16-experimental, or bf16-stable",
            );
        },
    }
    config.cuda.tuning.cache_directory = env::var_os("MIRMIR_CUDA_TUNING_CACHE").map(Into::into);
    if env::var_os("MIRMIR_SEMANTIC_DISABLE_TUNING").is_some() {
        config.cuda.tuning.mode = runtime::tuning::TuningMode::Disabled;
    }
    Ok(config)
}

fn argument(index: usize, name: &'static str) -> Result<PathBuf, Error> {
    env::args_os()
        .nth(index)
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment(name))
}
