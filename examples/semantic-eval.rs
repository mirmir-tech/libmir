#![allow(clippy::print_stdout)]

use std::{
    env,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

#[cfg(feature = "cuda")]
use cuda::{CudaKernelAdmission, CudaNumericalPolicy, CudaOutputHeadPolicy};
use libmir::{
    Conversation, Error, GenerationOverrides, GenerationRequest, Library, Message,
    ReasoningCyclePolicy, RuntimeConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    key: String,
    category: String,
    language: String,
    depth: usize,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

#[derive(Serialize)]
struct ResultRecord {
    key: String,
    category: String,
    language: String,
    depth: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    finish_reason: &'static str,
    text: String,
    reasoning: String,
    token_ids: Vec<u32>,
    metrics: runtime::metrics::GenerationMetrics,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = argument(1, "model path")?;
    let cases_path = argument(2, "cases JSONL path")?;
    let mut results = result_writer(env::args_os().nth(3).map(PathBuf::from))?;
    let library = Library::new(runtime_config()?);
    let model = library.load(&model_path, GenerationOverrides::default(), &mut |_| {})?;
    let clear_prefix_cache = env::var_os("MIRMIR_SEMANTIC_CLEAR_PREFIX_CACHE").is_some();
    for line in BufReader::new(File::open(cases_path)?).lines() {
        let case: Case = serde_json::from_str(&line?)?;
        if clear_prefix_cache {
            model.engine().clear_prefix_cache(model.handle())?;
        }
        let output = model.generate(&request(&case), &mut |_| {}, &mut |_| {})?;
        let record = ResultRecord {
            key: case.key,
            category: case.category,
            language: case.language,
            depth: case.depth,
            prompt_tokens: output.prompt_tokens,
            generated_tokens: output.token_ids.len(),
            finish_reason: output.finish_reason,
            text: output.text,
            reasoning: output.reasoning,
            token_ids: output.token_ids,
            metrics: output.metrics,
        };
        writeln!(results, "{}", serde_json::to_string(&record)?)?;
    }
    model.unload()?;
    Ok(())
}

fn result_writer(path: Option<PathBuf>) -> std::io::Result<BufWriter<Box<dyn Write>>> {
    let writer: Box<dyn Write> = match path {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(std::io::stdout()),
    };
    Ok(BufWriter::new(writer))
}

fn request(case: &Case) -> GenerationRequest {
    GenerationRequest {
        conversation: Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: case.prompt.clone(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: libmir::ToolChoice::default(),
        },
        options: GenerationOverrides {
            max_tokens: Some(case.max_tokens),
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(0),
            repetition_penalty: Some(repetition_penalty()),
            ..GenerationOverrides::default()
        },
        seed: Some(20_260_805),
        reasoning_cycle: if env::var_os("MIRMIR_SEMANTIC_HARMONY_EXIT").is_some() {
            ReasoningCyclePolicy::ExitReasoning {
                min_tokens: env::var("MIRMIR_SEMANTIC_HARMONY_EXIT_MIN_TOKENS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(192),
            }
        } else {
            ReasoningCyclePolicy::Recover
        },
    }
}

fn repetition_penalty() -> f32 {
    env::var("MIRMIR_SEMANTIC_REPETITION_PENALTY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0)
}

fn runtime_config() -> Result<RuntimeConfig, &'static str> {
    let mut config = RuntimeConfig::default();
    config.scheduler.max_batch_requests = 1;
    config.scheduler.decode_batch_wait_us = 0;
    #[cfg(feature = "cuda")]
    {
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
        config.cuda.tuning.cache_directory =
            env::var_os("MIRMIR_CUDA_TUNING_CACHE").map(Into::into);
        if env::var_os("MIRMIR_SEMANTIC_DISABLE_TUNING").is_some() {
            config.cuda.tuning.mode = runtime::tuning::TuningMode::Disabled;
        }
    }
    Ok(config)
}

fn argument(index: usize, name: &'static str) -> Result<PathBuf, Error> {
    env::args_os()
        .nth(index)
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment(name))
}

const fn default_max_tokens() -> usize {
    512
}
