use std::time::Instant;

use models::vision::{PixelShufflePreprocessedImage, PixelShufflePromptTokens};
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
    pub fn prefill_pixel_shuffle_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &PixelShufflePromptTokens,
        image: &PixelShufflePreprocessedImage,
        block_table: &BlockTable,
        sampling_logits: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<PrefillOutput> {
        let started = Instant::now();
        let lookup = model.id.clone();
        let model_id = lookup.clone();
        let prompt = prompt.clone();
        let image = image.clone();
        let blocks = block_table.blocks().to_vec();
        let execution_sampling = execution_sampling(
            sampling_logits.clone(),
            self.config.fusion.device_token_pipeline.enabled(),
        );
        Ok(self.with_model_progress(
            &lookup,
            move |loaded, worker_progress| {
                execute(
                    loaded,
                    &model_id,
                    session_id,
                    &prompt,
                    &image,
                    &blocks,
                    &sampling_logits,
                    &execution_sampling,
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
    model_id: &foundation::model::ModelId,
    session_id: Uuid,
    prompt: &PixelShufflePromptTokens,
    image: &PixelShufflePreprocessedImage,
    blocks: &[runtime::kv::BlockId],
    sampling_logits: &SamplingLogits,
    execution_sampling: &SamplingLogits,
    started: Instant,
    progress: &mut dyn FnMut(MetalProgressEvent),
) -> Result<PrefillOutput> {
    let native = loaded.prefill_pixel_shuffle_vision(
        session_id,
        prompt,
        image,
        blocks,
        execution_sampling,
        progress,
    )?;
    let output = output::materialize(loaded, native.output, sampling_logits)?;
    let cached_tokens = loaded.session_cached_tokens(session_id)?;
    let trace = format!(
        "native pixel-shuffle vision prefill: {} tokens, prefix cache disabled, {} runtime KV blocks, {} cached tokens, {:.3}ms",
        prompt.token_ids.len(),
        blocks.len(),
        cached_tokens,
        started.elapsed().as_secs_f64() * 1_000.0
    );
    tracing::debug!(
        model_id = %model_id,
        session_id = %session_id,
        prompt_tokens = prompt.token_ids.len(),
        cached_tokens,
        "native pixel-shuffle vision Metal prefill completed"
    );
    Ok(PrefillOutput {
        accepted_tokens: prompt.token_ids.len(),
        next_token: output.next_token,
        trace: Some(trace),
        logits: output.logits,
        candidates: output.candidates,
        timings: None,
    })
}
