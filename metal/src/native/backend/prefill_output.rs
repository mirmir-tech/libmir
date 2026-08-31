use std::time::Instant;

use runtime::backend::{PrefillOutput, PrefillRequest, SamplingLogits};
use uuid::Uuid;

use crate::native::{error::Result, model::LoadedModel, output, prefill::NativePrefill};

pub(super) fn materialize_prefill(
    loaded: &LoadedModel,
    request: &PrefillRequest,
    native: NativePrefill,
    started: Instant,
) -> Result<PrefillOutput> {
    materialize_prefill_parts(
        loaded,
        &request.model.id,
        request.session_id,
        &request.prompt_tokens,
        request.block_table.blocks().len(),
        request.sampling_logits,
        native,
        started,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_prefill_parts(
    loaded: &LoadedModel,
    model_id: &str,
    session_id: Uuid,
    prompt_tokens: &[u32],
    blocks: usize,
    sampling_logits: SamplingLogits,
    native: NativePrefill,
    started: Instant,
) -> Result<PrefillOutput> {
    let prefix_cache_tokens = native.prefix_cache_tokens;
    let output = output::materialize(loaded, native.output, sampling_logits)?;
    let cached_tokens = loaded.session_cached_tokens(session_id)?;
    let prefix_cache = if prefix_cache_tokens > 0 {
        format!("device prefix hit for {prefix_cache_tokens} tokens")
    } else {
        "device prefix miss".into()
    };
    let trace = format!(
        "native prefill: {} tokens, {}, {} runtime KV blocks, {} cached tokens, {:.3}ms",
        prompt_tokens.len(),
        prefix_cache,
        blocks,
        cached_tokens,
        started.elapsed().as_secs_f64() * 1000.0
    );
    tracing::debug!(
        model_id,
        session_id = %session_id,
        prompt_tokens = prompt_tokens.len(),
        cached_tokens,
        prefix_cache_tokens,
        "native MLX prefill completed"
    );
    Ok(PrefillOutput {
        accepted_tokens: prompt_tokens.len(),
        next_token: output.next_token,
        trace: Some(trace),
        logits: output.logits,
        candidates: output.candidates,
        timings: None,
    })
}
