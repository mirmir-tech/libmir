//! Generates greedy continuations of reference prompts through the public
//! generation path, with or without speculative decoding, and compares the
//! tokens with an earlier run.
//!
//! Usage: `speculative-generate <model> <reference.json> [options]`, where
//! the reference is written by `benchmarks/tools/nemotron/greedy_oracle.py`.
//! Options: `draft=K` drafts `K` tokens per step with the checkpoint's
//! prediction module; `tokens=N` caps each continuation; `temperature=T`
//! samples with top-p 0.95 instead of greedy decoding; `rows=R` generates
//! the first `R` prompts at once and reports their aggregate rate; `out=PATH`
//! writes the token ids; `compare=PATH` reports where they part from that
//! file's.
#![allow(clippy::print_stdout)]

use std::{env, fs, num::NonZeroUsize, path::PathBuf};

use libmir::{
    Conversation, GenerationOverrides, GenerationRequest, Library, Message, RuntimeConfig,
    Speculation,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Reference {
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    prompt: String,
}

#[derive(Serialize, Deserialize)]
struct Run {
    token_ids: Vec<Vec<u32>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let model = args.next().map(PathBuf::from).ok_or("missing model path")?;
    let reference = args.next().map(PathBuf::from).ok_or("missing reference path")?;
    let reference: Reference = serde_json::from_slice(&fs::read(reference)?)?;
    let mut config = RuntimeConfig::default();
    let (mut tokens, mut out, mut compare, mut temperature) = (256, None, None, 0.0);
    let mut rows = None;
    for option in args.filter_map(|option| option.into_string().ok()) {
        if let Some(draft) = option.strip_prefix("draft=") {
            let draft_tokens = NonZeroUsize::new(draft.parse()?).ok_or("draft=0")?;
            config.scheduler.speculation = Some(Speculation { draft_tokens });
        } else if let Some(count) = option.strip_prefix("tokens=") {
            tokens = count.parse()?;
        } else if let Some(value) = option.strip_prefix("rows=") {
            rows = Some(value.parse::<usize>()?);
        } else if let Some(value) = option.strip_prefix("temperature=") {
            temperature = value.parse()?;
        } else if let Some(path) = option.strip_prefix("out=") {
            out = Some(PathBuf::from(path));
        } else if let Some(path) = option.strip_prefix("compare=") {
            compare = Some(serde_json::from_slice::<Run>(&fs::read(path)?)?);
        } else {
            return Err(format!("unknown option {option}").into());
        }
    }
    let library = Library::new(config);
    let model = library.load(model, GenerationOverrides::default(), &mut |_| {})?;
    if let Some(rows) = rows {
        concurrent(&model, &reference.records[..rows], tokens, temperature)?;
        model.unload()?;
        return Ok(());
    }
    let mut run = Run { token_ids: Vec::new() };
    for (index, record) in reference.records.iter().enumerate() {
        let request = request(&record.prompt, tokens, temperature);
        let output = model.generate(&request, &mut |_| {}, &mut |_| {})?;
        let divergence = compare.as_ref().and_then(|other| {
            let other = other.token_ids.get(index)?;
            Some(output.token_ids.iter().zip(other).position(|(a, b)| a != b))
        });
        let counts = &output.metrics.tokens;
        println!(
            "prompt={index} tokens={} decode_tok_s={:.2} tokens_per_step={:.3} \
             first_divergence={divergence:?}",
            output.token_ids.len(),
            output.metrics.throughput.decode.per_second.unwrap_or(f64::NAN),
            ratio(counts.decode_tokens, counts.decode_steps.max(1)),
        );
        if temperature > 0.0 {
            println!("  text={:?}", output.text.chars().take(160).collect::<String>());
        }
        run.token_ids.push(output.token_ids);
    }
    if let Some(path) = out {
        fs::write(path, serde_json::to_vec(&run)?)?;
    }
    model.unload()?;
    Ok(())
}

/// Generates every record at once and reports the aggregate decode rate.
fn concurrent(
    model: &libmir::Model,
    records: &[Record],
    tokens: usize,
    temperature: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = std::time::Instant::now();
    let generated = std::thread::scope(|scope| {
        // Every generation starts before the first is joined.
        let mut workers = Vec::with_capacity(records.len());
        for record in records {
            let request = request(&record.prompt, tokens, temperature);
            workers.push(scope.spawn(move || model.generate(&request, &mut |_| {}, &mut |_| {})));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().map_err(|_| "generation thread panicked"))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let total = generated
        .into_iter()
        .map(|output| Ok::<_, libmir::Error>(output?.token_ids.len()))
        .sum::<Result<usize, _>>()?;
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "rows={} generated={total} seconds={seconds:.2} aggregate_tok_s={:.2}",
        records.len(),
        ratio(total, 1) / seconds
    );
    Ok(())
}

fn request(prompt: &str, tokens: usize, temperature: f32) -> GenerationRequest {
    GenerationRequest {
        tool_constraints: libmir::ToolConstraints::None,
        reasoning_token_budget: None,
        conversation: Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: prompt.into(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: libmir::ToolChoice::default(),
        },
        options: GenerationOverrides {
            max_tokens: Some(tokens),
            temperature: Some(temperature),
            top_p: Some(if temperature > 0.0 {
                0.95
            } else {
                1.0
            }),
            top_k: Some(0),
            repetition_penalty: Some(1.0),
            ..GenerationOverrides::default()
        },
        seed: Some(7),
        reasoning_cycle: libmir::ReasoningCyclePolicy::default(),
        reasoning: libmir::ReasoningMode::ModelDefault,
        reasoning_effort: libmir::ReasoningEffort::ModelDefault,
        reasoning_history: libmir::ReasoningHistory::ModelDefault,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    let count = |value: usize| f64::from(u32::try_from(value).unwrap_or(u32::MAX));
    count(numerator) / count(denominator)
}
