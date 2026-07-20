use std::time::Instant;

use models::vision::{SpatialMergePreprocessedImage, SpatialMergePromptTokens};
use runtime::{
    Result as RuntimeResult,
    backend::{ModelHandle, PrefillOutput, SamplingLogits},
    kv::BlockTable,
};
use uuid::Uuid;

use super::super::{MetalBackend, execution::execution_sampling};
use crate::{
    MetalProgressEvent,
    native::{error::Result, model::LoadedModel, output},
};

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_spatial_merge_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &SpatialMergePromptTokens,
        image: &SpatialMergePreprocessedImage,
        block_table: &BlockTable,
        sampling_logits: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        let started = Instant::now();
        let lookup = model.id.clone();
        let model_id = lookup.clone();
        let prompt = prompt.clone();
        let image = image.clone();
        let blocks = block_table.blocks().len();
        let execution_sampling =
            execution_sampling(sampling_logits, self.config.fusion.device_token_pipeline.enabled());
        Ok(self.with_model_progress(
            &lookup,
            move |loaded, worker_progress| {
                execute(
                    loaded,
                    &model_id,
                    session_id,
                    &prompt,
                    &image,
                    blocks,
                    sampling_logits,
                    execution_sampling,
                    started,
                    worker_progress,
                )
            },
            progress,
        )?)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute(
    loaded: &mut LoadedModel,
    model_id: &str,
    session_id: Uuid,
    prompt: &SpatialMergePromptTokens,
    image: &SpatialMergePreprocessedImage,
    blocks: usize,
    sampling_logits: SamplingLogits,
    execution_sampling: SamplingLogits,
    started: Instant,
    progress: &mut dyn FnMut(MetalProgressEvent),
) -> Result<PrefillOutput> {
    let native = loaded.prefill_spatial_merge_vision(
        session_id,
        prompt,
        image,
        execution_sampling,
        progress,
    )?;
    let output = output::materialize(loaded, native.output, sampling_logits)?;
    let cached_tokens = loaded.session_cached_tokens(session_id)?;
    let trace = format!(
        "native spatial-merge vision prefill: {} tokens, prefix cache disabled, {} runtime KV blocks, {} cached tokens, {:.3}ms",
        prompt.token_ids.len(),
        blocks,
        cached_tokens,
        started.elapsed().as_secs_f64() * 1_000.0
    );
    tracing::debug!(
        model_id,
        session_id = %session_id,
        prompt_tokens = prompt.token_ids.len(),
        cached_tokens,
        mrope_delta = prompt.position_delta,
        "native spatial-merge vision Metal prefill completed"
    );
    Ok(PrefillOutput {
        accepted_tokens: prompt.token_ids.len(),
        next_token: output.next_token,
        trace: Some(trace),
        logits: output.logits,
        candidates: output.candidates,
    })
}
