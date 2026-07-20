use std::{cmp::Ordering, path::PathBuf};

use libmir::{
    ChatCompletionRequest, ChatMessage, GenerationOverrides, IMAGE_PLACEHOLDER, Library, Result,
    RuntimeConfig, SamplingLogits,
};
use runtime::backend::LogitsTrace;

const BLACK_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 4, 0,
    0, 0, 181, 28, 12, 2, 0, 0, 0, 11, 73, 68, 65, 84, 120, 218, 99, 100, 248, 15, 0, 1, 5, 1, 1,
    39, 24, 227, 102, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const GENERATED_TOKENS: usize = 32;

#[test]
#[ignore = "loads a real vision checkpoint; set MODEL and LIBMIR_VISION_COMPARISON_REPORT"]
fn records_vision_prefill_and_greedy_decode() -> Result<()> {
    let model_path = required_path("MODEL")?;
    let report_path = required_path("LIBMIR_VISION_COMPARISON_REPORT")?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 128;
    let model =
        Library::new(config).load(model_path, GenerationOverrides::default(), &mut |_event| {})?;
    let request = request(&model);
    let prepared = model.prepare_image(&request, BLACK_PNG)?;
    let mut session = model.session();
    let prefill = session.prefill_vision(&prepared, SamplingLogits::Full, &mut |_event| {})?;
    let logits = prefill.logits.as_ref().ok_or_else(|| {
        runtime::RuntimeError::Backend("vision prefill did not return full logits".into())
    })?;
    let first = argmax(logits)?;
    let mut token_ids = vec![first];
    while token_ids.len() < GENERATED_TOKENS {
        let current = *token_ids.last().ok_or_else(|| {
            runtime::RuntimeError::Backend("comparison token sequence is empty".into())
        })?;
        let next = session
            .decode(current, SamplingLogits::None)?
            .event
            .token_id
            .ok_or_else(|| runtime::RuntimeError::Backend("decode returned no token".into()))?;
        token_ids.push(next);
    }
    let report = report(&model, &prefill, logits, &token_ids);
    std::fs::write(&report_path, report).map_err(|error| {
        runtime::RuntimeError::Backend(format!(
            "failed to write comparison report {}: {error}",
            report_path.display()
        ))
    })?;
    Ok(())
}

fn required_path(name: &'static str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or(libmir::Error::MissingEnvironment(name))
}

fn request(model: &libmir::Model) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.handle().id.clone(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: format!("{IMAGE_PLACEHOLDER}Describe the image in one short sentence."),
            reasoning_content: None,
        }],
        stream: false,
        max_tokens: Some(GENERATED_TOKENS),
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(7),
    }
}

fn argmax(logits: &LogitsTrace) -> Result<u32> {
    logits
        .values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .map(u32::try_from)
        .transpose()
        .map_err(|error| runtime::RuntimeError::Backend(error.to_string()))?
        .ok_or_else(|| {
            runtime::RuntimeError::Backend("prefill logits contain no finite value".into()).into()
        })
}

fn report(
    model: &libmir::Model,
    prefill: &libmir::PrefillOutput,
    logits: &LogitsTrace,
    token_ids: &[u32],
) -> String {
    let mut ranked = logits
        .values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    ranked.truncate(16);
    let (sum, sum_abs, sum_squares) = logits.values.iter().filter(|value| value.is_finite()).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(sum, sum_abs, sum_squares), value| {
            let value = f64::from(*value);
            (sum + value, sum_abs + value.abs(), value.mul_add(value, sum_squares))
        },
    );
    format!(
        "backend={}\naccepted_tokens={}\nlogits_shape={:?}\nfinite={}\n\
         non_finite={}\nmin_max={:?}\nsum={sum:.9}\nsum_abs={sum_abs:.9}\n\
         sum_squares={sum_squares:.9}\ntop16={ranked:?}\ntoken_ids={token_ids:?}\n",
        model.handle().backend,
        prefill.accepted_tokens,
        logits.shape,
        logits.finite_count(),
        logits.non_finite_count(),
        logits.finite_min_max(),
    )
}
