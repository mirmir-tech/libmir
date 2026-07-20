pub(super) mod model;

use models::{
    layout::VisionPipeline,
    vision::{
        PooledPreprocessedImage, PooledPromptTokens, SpatialMergePreprocessedImage,
        SpatialMergePromptTokens,
    },
};
use runtime::{
    backend::{ModelHandle, PrefillOutput, SamplingLogits},
    kv::BlockTable,
    progress::ProgressEvent,
};
use uuid::Uuid;

use super::{
    CudaEngine,
    model::{DeviceToken, LoadedModel, ModelExecution},
};
use crate::{Error, Result};

impl CudaEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_pooled_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &PooledPromptTokens,
        image: &PooledPreprocessedImage,
        block_table: &BlockTable,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let loaded = self.model(&model.id)?;
        let vision = require_vision(&loaded, VisionPipeline::PooledEncoder)?;
        let model::LoadedVisionModel::Pooled(tower) = vision else {
            return Err(Error::UnsupportedVisionContract(
                "loaded CUDA vision tower differs from the discovered pooled contract".into(),
            ));
        };
        if prompt.image_start >= prompt.image_end
            || prompt.image_end > prompt.token_ids.len()
            || prompt.image_end - prompt.image_start != image.soft_tokens
        {
            return Err(Error::InvalidVisionKernel(
                "pooled prompt image span differs from preprocessing",
            ));
        }
        progress(ProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let encoded = tower
            .forward_preprocessed_scheduled(image, &mut |step| submit_vision_step(&loaded, step))?;
        let expected_elements = image
            .soft_tokens
            .checked_mul(loaded.decoder.hidden_size)
            .ok_or(Error::InvalidVisionKernel("spatial-merge output size overflow"))?;
        if encoded.tokens != image.soft_tokens
            || encoded.width != loaded.decoder.hidden_size
            || encoded.hidden.len() != expected_elements
        {
            return Err(Error::InvalidVisionKernel(
                "pooled tower output differs from decoder embedding contract",
            ));
        }
        let mut runner = loaded.prefill_runner()?;
        let ModelExecution::Standard(model) = &mut runner.execution else {
            return Err(Error::UnsupportedVisionContract(
                "pooled vision requires the standard CUDA decoder path".into(),
            ));
        };
        let model = model.as_mut();
        model.prefill_vision_for_sampling(
            session_id,
            &prompt.token_ids,
            &encoded.hidden,
            prompt.image_start,
            prompt.image_end,
            tower.bidirectional_image_attention(),
            block_table,
            sampling,
        )?;
        drop(encoded);
        let completed = self.output(model, sampling)?;
        runner.selected = completed.token.map(|token| DeviceToken { session: session_id, token });
        drop(runner);
        loaded.register_session(session_id)?;
        progress(ProgressEvent::prefill_tokens(prompt.token_ids.len(), prompt.token_ids.len()));
        Ok(PrefillOutput {
            accepted_tokens: prompt.token_ids.len(),
            next_token: completed.token,
            trace: Some(format!(
                "cuda.pooled-vision=device-tower+mixed-paged-prefill;layers={}",
                tower.layer_count()
            )),
            logits: completed.logits,
            candidates: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_spatial_merge_vision_with_progress(
        &self,
        model: &ModelHandle,
        session_id: Uuid,
        prompt: &SpatialMergePromptTokens,
        image: &SpatialMergePreprocessedImage,
        block_table: &BlockTable,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let loaded = self.model(&model.id)?;
        let vision = require_vision(&loaded, VisionPipeline::SpatialMergeEncoder)?;
        let model::LoadedVisionModel::SpatialMerge(tower) = vision else {
            return Err(Error::UnsupportedVisionContract(
                "loaded CUDA vision tower differs from the discovered spatial-merge contract"
                    .into(),
            ));
        };
        if prompt.image_start >= prompt.image_end
            || prompt.image_end > prompt.token_ids.len()
            || prompt.image_end - prompt.image_start != image.soft_tokens
        {
            return Err(Error::InvalidVisionKernel(
                "spatial-merge prompt image span differs from preprocessing",
            ));
        }
        progress(ProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let encoded = tower
            .forward_preprocessed_scheduled(image, &mut |step| submit_vision_step(&loaded, step))?;
        let expected_elements = image
            .soft_tokens
            .checked_mul(loaded.decoder.hidden_size)
            .ok_or(Error::InvalidVisionKernel("spatial-merge output size overflow"))?;
        if encoded.tokens != image.soft_tokens
            || encoded.width != loaded.decoder.hidden_size
            || encoded.hidden.len() != expected_elements
        {
            return Err(Error::InvalidVisionKernel(
                "spatial-merge tower output differs from decoder embedding contract",
            ));
        }
        let mut runner = loaded.prefill_runner()?;
        let ModelExecution::Hybrid(hybrid) = &mut runner.execution else {
            return Err(Error::UnsupportedVisionContract(
                "spatial-merge vision requires a hybrid CUDA decoder".into(),
            ));
        };
        let mut session = hybrid.template.instantiate()?;
        session.prefill_vision(
            session_id,
            &prompt.token_ids,
            &prompt.position_ids,
            block_table,
            (prompt.image_start, prompt.image_end),
            &encoded.hidden,
            prompt.position_delta,
        )?;
        drop(encoded);
        let completed = self.output_hybrid(&mut session, sampling)?;
        hybrid.sessions.insert(session_id, session);
        runner.selected = completed.token.map(|token| DeviceToken { session: session_id, token });
        drop(runner);
        loaded.register_session(session_id)?;
        progress(ProgressEvent::prefill_tokens(prompt.token_ids.len(), prompt.token_ids.len()));
        Ok(PrefillOutput {
            accepted_tokens: prompt.token_ids.len(),
            next_token: completed.token,
            trace: Some(format!(
                "cuda.spatial-merge=device-tower+hybrid-mixed-prefill;layers={}",
                tower.layer_count()
            )),
            logits: completed.logits,
            candidates: None,
        })
    }
}

fn submit_vision_step(model: &LoadedModel, step: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    // The guard serializes shared-stream submission. Dropping it after one phase
    // lets a waiting decode enter before the next layer without synchronizing CUDA.
    let _runner = model.prefill_runner()?;
    step()
}

fn require_vision(
    model: &LoadedModel,
    expected: VisionPipeline,
) -> Result<&model::LoadedVisionModel> {
    let Some(config) = model.vision.as_ref() else {
        return Err(Error::UnsupportedVisionContract(
            "checkpoint has no discovered vision execution contract".into(),
        ));
    };
    if config.pipeline() != expected {
        return Err(Error::UnsupportedVisionContract(format!(
            "request requires {expected:?}, checkpoint declares {:?}",
            config.pipeline()
        )));
    }
    let readiness = model.vision_readiness.as_ref().ok_or_else(|| {
        Error::UnsupportedVisionContract("checkpoint vision readiness is unavailable".into())
    })?;
    if !readiness.is_ready() {
        return Err(Error::UnsupportedVisionContract(format!(
            "image execution contract is incomplete: {} of {} tensors are present",
            readiness.present, readiness.required
        )));
    }
    model.vision_model.as_ref().ok_or_else(|| {
        Error::UnsupportedVisionContract(
            "vision tensors are ready but the CUDA tower is not loaded".into(),
        )
    })
}
