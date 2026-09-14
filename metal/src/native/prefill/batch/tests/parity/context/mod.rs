mod metrics;
mod prompts;

use std::io::Write;

use models::{layout::ModelLayout, tokenizer::TextTokenizer};

use super::super::{LoadedModel, Result, fixture};
use crate::{
    config::RouterPrecision,
    native::{error::Error, prefill::evaluation, session::SessionState, step},
};

#[test]
#[ignore = "real Qwen matched-chunk long-context routing, logits and retrieval answers"]
fn compares_router_precision_context() -> Result<()> {
    let path = std::env::var_os("MIRMIR_BENCH_MODEL")
        .ok_or_else(|| Error::Benchmark("set MIRMIR_BENCH_MODEL".into()))?;
    let path = std::path::PathBuf::from(path);
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&path)?)?;
    let mut model = fixture::load_path_with_context(path.to_string_lossy().into_owned(), 0, 2080)?;
    let mut wrong = 0;
    for context in [512, 2048] {
        let prompts = prompts::build(&tokenizer, context)?;
        for chunk in [128, 512] {
            for precision in [RouterPrecision::Model, RouterPrecision::Float32] {
                model.stream.synchronize()?;
                model.stream.set_router_precision_for_test(precision);
                wrong += compare(&mut model, &prompts, &tokenizer, chunk, precision)?;
            }
        }
    }
    assert_eq!(wrong, 0, "retrieval errors retained in per-case records");
    Ok(())
}

fn states(model: &LoadedModel, context: usize) -> Result<Vec<SessionState>> {
    (0..5)
        .map(|_| {
            let mut cache = model.execution.decoder()?.new_cache(model.stream())?;
            cache.reserve(context + 32)?;
            cache.plan_contiguous(context + 32);
            Ok(SessionState::new(cache))
        })
        .collect()
}

fn compare(
    model: &mut LoadedModel,
    prompts: &[Vec<u32>],
    tokenizer: &TextTokenizer,
    chunk: usize,
    precision: RouterPrecision,
) -> Result<usize> {
    let context = prompts[0].len();
    assert!(prompts.iter().all(|p| p.len() == context));
    let mut scalar = states(model, context)?;
    let mut packed = states(model, context)?;
    model.reserve_prefill_pages(10 * (context + 32).div_ceil(16))?;
    let mut position = 0;
    while position < context - 1 {
        let count = chunk.min(context - 1 - position);
        let mut scalar_hidden = Vec::new();
        for (state, prompt) in scalar.iter_mut().zip(prompts) {
            let hidden = step::forward_prefill_state(
                model.execution.decoder()?,
                model.stream(),
                state,
                &prompt[position..position + count],
                position,
            )?;
            evaluation::materialize(model, state, &hidden)?;
            scalar_hidden.extend(hidden.to_vec_f32(model.stream())?);
            assert_eq!(state.cache.cached_tokens()?, position + count);
        }
        let tokens = prompts
            .iter()
            .flat_map(|p| p[position..position + count].iter().copied())
            .collect::<Vec<_>>();
        let mut refs = packed.iter_mut().collect::<Vec<_>>();
        let hidden = step::forward_packed_prefill_state(
            model.execution.decoder()?,
            model.stream(),
            &mut refs,
            &[position; 5],
            &tokens,
            count,
        )?;
        evaluation::materialize_packed(model, &refs, &hidden)?;
        if position == 0 || position + count == context - 1 {
            let mut state_metrics = Vec::new();
            for layer in [0, 20] {
                let a = scalar[0]
                    .cache
                    .gated_delta_state(layer)?
                    .values()?
                    .to_vec_f32(model.stream())?;
                let b = packed[0]
                    .cache
                    .gated_delta_state(layer)?
                    .values()?
                    .to_vec_f32(model.stream())?;
                state_metrics.push(
                    serde_json::json!({"layer":layer,"difference":metrics::difference(&a,&b)}),
                );
            }
            writeln!(
                std::io::stderr().lock(),
                "router.context.state: {}",
                serde_json::json!({
                    "context":context,"chunk":chunk,"precision":format!("{precision:?}"),"offset":position+count,
                    "hidden":metrics::difference(&scalar_hidden,&hidden.to_vec_f32(model.stream())?),"states":state_metrics
                })
            )?;
        }
        for state in &packed {
            assert_eq!(state.cache.cached_tokens()?, position + count);
        }
        position += count;
    }
    let mut wrong = 0;
    for (row, ((a, b), prompt)) in scalar.iter_mut().zip(&mut packed).zip(prompts).enumerate() {
        let (a_logits, a_answer, a_eos) = answer(model, a, prompt, tokenizer)?;
        let (b_logits, b_answer, b_eos) = answer(model, b, prompt, tokenizer)?;
        let expected = prompts::ANSWERS[row];
        wrong += usize::from(!a_eos || a_answer.trim() != expected);
        wrong += usize::from(!b_eos || b_answer.trim() != expected);
        writeln!(
            std::io::stderr().lock(),
            "router.context.answer: {}",
            serde_json::json!({
                "context":context,"chunk":chunk,"precision":format!("{precision:?}"),"row":row,
                "logits":metrics::logits(&a_logits,&b_logits),"scalar_answer":a_answer,"packed_answer":b_answer,
                "scalar_eos":a_eos,"packed_eos":b_eos,"expected":expected
            })
        )?;
    }
    drop((scalar, packed));
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(wrong)
}

fn answer(
    model: &LoadedModel,
    state: &mut SessionState,
    prompt: &[u32],
    tokenizer: &TextTokenizer,
) -> Result<(Vec<f32>, String, bool)> {
    let mut token = prompt[prompt.len() - 1];
    let mut tokens = Vec::new();
    let mut first_logits = Vec::new();
    for step_index in 0..16 {
        let output = step::forward_token(
            model.execution.decoder()?,
            model.stream(),
            state,
            token,
            prompt.len() - 1 + step_index,
            false,
        )?;
        evaluation::materialize(model, state, &output)?;
        let logits = output.to_vec_f32(model.stream())?;
        token = u32::try_from(metrics::argmax(&logits))?;
        if step_index == 0 {
            first_logits = logits;
        }
        if tokenizer.stop_token_ids().contains(&token) {
            return Ok((first_logits, tokenizer.decode(&tokens)?, true));
        }
        tokens.push(token);
    }
    Ok((first_logits, tokenizer.decode(&tokens)?, false))
}
