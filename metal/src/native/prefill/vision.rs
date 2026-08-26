use models::vision::{
    PooledPreprocessedImage, PooledPromptTokens, SpatialMergePreprocessedImage,
    SpatialMergePromptTokens,
};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::NativePrefill;
use crate::{
    MetalProgressEvent,
    native::{
        error::{Error, Result},
        model::{LoadedModel, LoadedVisionModel},
        session::SessionState,
        step,
    },
};

impl LoadedModel {
    pub(in crate::native) fn prefill_pooled_vision(
        &mut self,
        session: Uuid,
        prompt: &PooledPromptTokens,
        image: &PooledPreprocessedImage,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let Some((&last, prefix)) = prompt.token_ids.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        if prompt.image_end > prefix.len() {
            return Err(Error::UnsupportedModel(
                "pooled vision image block must precede the final prompt token".into(),
            ));
        }
        let mut state = {
            let model = self.execution.decoder()?;
            SessionState::new(model.new_cache(&self.stream)?)
        };
        let reserve = prompt.token_ids.len().max(self.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        state.cache.plan_contiguous(prompt.token_ids.len().saturating_add(page_size));
        self.reserve_prefill_pages(super::required_prefill_pages(
            prompt.token_ids.len(),
            0,
            page_size,
        ))?;
        let model = self.execution.decoder()?;
        let Some(LoadedVisionModel::PooledEncoder(tower)) = self.vision_model.as_ref() else {
            return Err(Error::UnsupportedModel(
                "pooled vision tower is not loaded or its tensors are incomplete".into(),
            ));
        };
        progress(MetalProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let prefix_prompt = PooledPromptTokens {
            token_ids: prefix.to_vec(),
            image_start: prompt.image_start,
            image_end: prompt.image_end,
        };
        let hidden = tower.forward_multimodal_prefill(
            model, &prefix_prompt, image, &mut state.cache, &self.stream,
        )?;
        hidden.async_eval(&self.stream)?;
        self.settle_prefill_graph()?;
        state.cache.detach_evaluated_graphs(&self.stream)?;
        progress(MetalProgressEvent::prefill_tokens(prefix.len(), prompt.token_ids.len()));
        let logits = step::forward_token(
            model,
            &self.stream,
            &mut state,
            last,
            prefix.len(),
            sampling == SamplingLogits::None,
        )?;
        state.position = prompt.token_ids.len();
        let output = step::output(model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        progress(MetalProgressEvent::prefill_tokens(
            prompt.token_ids.len(),
            prompt.token_ids.len(),
        ));
        Ok(NativePrefill { output, prefix_cache_tokens: 0 })
    }

    pub(in crate::native) fn prefill_spatial_merge_vision(
        &mut self,
        session: Uuid,
        prompt: &SpatialMergePromptTokens,
        image: &SpatialMergePreprocessedImage,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let Some((&last, prefix)) = prompt.token_ids.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        if prompt.image_end > prefix.len() {
            return Err(Error::UnsupportedModel(
                "spatial-merge vision image block must precede the final prompt token".into(),
            ));
        }
        let mut state = {
            let model = self.execution.decoder()?;
            SessionState::new(model.new_cache(&self.stream)?)
        };
        state.rope_position_delta = prompt.position_delta;
        state
            .cache
            .reserve(prompt.token_ids.len().max(self.stream.config().cache.kv_reserve_tokens))?;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        state.cache.plan_contiguous(prompt.token_ids.len().saturating_add(page_size));
        self.reserve_prefill_pages(super::required_prefill_pages(
            prompt.token_ids.len(),
            0,
            page_size,
        ))?;
        let model = self.execution.decoder()?;
        let Some(LoadedVisionModel::SpatialMergeEncoder(tower)) = self.vision_model.as_ref() else {
            return Err(Error::UnsupportedModel(
                "spatial-merge vision tower is not loaded or its tensors are incomplete".into(),
            ));
        };
        progress(MetalProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let prefix_prompt = spatial_merge_prefix(prompt, prefix.len());
        let hidden = tower.forward_multimodal_prefill(
            model, &prefix_prompt, image, &mut state.cache, &self.stream,
        )?;
        hidden.async_eval(&self.stream)?;
        self.settle_prefill_graph()?;
        state.cache.detach_evaluated_graphs(&self.stream)?;
        progress(MetalProgressEvent::prefill_tokens(prefix.len(), prompt.token_ids.len()));
        state.position = prefix.len();
        let model_position = state.model_position()?;
        let logits = step::forward_token(
            model,
            &self.stream,
            &mut state,
            last,
            model_position,
            sampling == SamplingLogits::None,
        )?;
        state.position = prompt.token_ids.len();
        let output = step::output(model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        progress(MetalProgressEvent::prefill_tokens(
            prompt.token_ids.len(),
            prompt.token_ids.len(),
        ));
        Ok(NativePrefill { output, prefix_cache_tokens: 0 })
    }
}

fn spatial_merge_prefix(
    prompt: &SpatialMergePromptTokens,
    length: usize,
) -> SpatialMergePromptTokens {
    let sequence = prompt.token_ids.len();
    let mut position_ids = Vec::with_capacity(3 * length);
    for axis in 0..3 {
        let start = axis * sequence;
        position_ids.extend_from_slice(&prompt.position_ids[start..start + length]);
    }
    SpatialMergePromptTokens {
        token_ids: prompt.token_ids[..length].to_vec(),
        image_start: prompt.image_start,
        image_end: prompt.image_end,
        position_ids,
        position_delta: prompt.position_delta,
    }
}
