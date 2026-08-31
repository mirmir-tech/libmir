use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::BenchmarkConfig;
use crate::native::{
    error::Result,
    model::{LoadedModel, NativeOutput},
};

const CONTEXT_TOKENS: usize = 128;
const DECODE_TOKENS: usize = 128;
const PROMPTS: usize = 3;

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn preserves_multi_prompt_greedy_pipeline() -> Result<()> {
    let config = BenchmarkConfig {
        model: super::model_path()?,
        decode_tokens: DECODE_TOKENS,
        prompt_tokens: CONTEXT_TOKENS,
        samples: 1,
        warmup: 0,
    };
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load_with_config(
        &config.manifest(),
        super::diagnostics::isolated_config(),
        &mut ignored,
    )?;

    for seed in 0..PROMPTS {
        let prompt = prompt(seed)?;
        let reference = generate(&mut model, &prompt, SamplingLogits::Full)?;
        let pipelined = generate(&mut model, &prompt, SamplingLogits::None)?;
        assert_eq!(pipelined, reference, "greedy pipeline differs for prompt seed {seed}");
    }
    Ok(())
}

fn generate(model: &mut LoadedModel, prompt: &[u32], sampling: SamplingLogits) -> Result<Vec<u32>> {
    let session = Uuid::new_v4();
    let mut ignored = |_event| {};
    let output = model.prefill(session, prompt, &[], sampling, None, &mut ignored)?;
    let mut token = output_token(model, output.output)?;
    let mut tokens = Vec::with_capacity(DECODE_TOKENS + 1);
    tokens.push(token);
    for _ in 0..DECODE_TOKENS {
        let output = model.decode(session, token, sampling)?;
        token = output_token(model, output)?;
        tokens.push(token);
    }
    model.release_session(session)?;
    Ok(tokens)
}

fn output_token(model: &LoadedModel, output: NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(token),
        NativeOutput::Logits(logits) => Ok(logits.argmax_u32(model.stream())?),
    }
}

fn prompt(seed: usize) -> Result<Vec<u32>> {
    (0..CONTEXT_TOKENS)
        .map(|index| Ok(u32::try_from((seed * 1_009 + index) % 100_000 + 1_000)?))
        .collect()
}
