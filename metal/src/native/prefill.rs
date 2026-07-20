use models::vision::{
    PooledPreprocessedImage, PooledPromptTokens, SpatialMergePreprocessedImage,
    SpatialMergePromptTokens,
};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{
    error::{Error, Result},
    model::{LoadedModel, LoadedVisionModel, NativeOutput},
    session::SessionState,
    step,
};
use crate::MetalProgressEvent;

pub(super) struct NativePrefill {
    pub(super) output: NativeOutput,
    pub(super) prefix_cache_tokens: usize,
}

impl LoadedModel {
    pub(super) fn prefill(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let Some((&last, prefix)) = tokens.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        let restored = self.prefixes.restore_longest(&self.info.manifest.id, tokens)?;
        let (mut state, mut position, prefix_cache_tokens, cached_logits) =
            if let Some((state, logits)) = restored {
                let position = state.position;
                (state, position, position, Some(logits))
            } else {
                (SessionState::new(self.model.new_cache(&self.stream)?), 0, 0, None)
            };
        let reserve = tokens.len().max(self.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        if position == tokens.len() {
            let logits = cached_logits.ok_or(Error::NoPrefixLogits)?;
            let output = step::output(&self.model, &self.stream, &mut state, logits, sampling)?;
            self.sessions.insert(session, state);
            progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
            return Ok(NativePrefill { output, prefix_cache_tokens });
        }

        progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
        let mut remaining = &prefix[position..];
        while !remaining.is_empty() {
            let count = self.prefill_chunk_len(position, remaining.len());
            let logits = step::forward_prefill(
                &self.model,
                &self.stream,
                &mut state,
                &remaining[..count],
                position,
            )?;
            logits.async_eval()?;
            self.stream.synchronize()?;
            position += count;
            remaining = &remaining[count..];
            progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
        }
        let logits = step::forward_token(
            &self.model,
            &self.stream,
            &mut state,
            last,
            position,
            sampling == SamplingLogits::None,
        )?;
        state.position = tokens.len();
        self.prefixes.insert(&self.info.manifest.id, tokens, &state, &logits)?;
        progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
        let output = step::output(&self.model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        Ok(NativePrefill { output, prefix_cache_tokens })
    }

    pub(super) fn prefill_pooled_vision(
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
        let Some(LoadedVisionModel::PooledEncoder(tower)) = self.vision_model.as_ref() else {
            return Err(Error::UnsupportedModel(
                "pooled vision tower is not loaded or its tensors are incomplete".into(),
            ));
        };
        let crate::engine::DecoderModel::HybridMoe(decoder) = &self.model else {
            return Err(Error::UnsupportedModel(
                "pooled vision Metal multimodal prefill currently requires the hybrid MoE decoder"
                    .into(),
            ));
        };
        let mut state = SessionState::new(decoder.new_cache(&self.stream)?);
        let reserve = prompt.token_ids.len().max(self.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        progress(MetalProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let prefix_prompt = PooledPromptTokens {
            token_ids: prefix.to_vec(),
            image_start: prompt.image_start,
            image_end: prompt.image_end,
        };
        let hidden = tower.forward_multimodal_prefill(
            decoder, &prefix_prompt, image, &mut state.cache, &self.stream,
        )?;
        hidden.async_eval()?;
        self.stream.synchronize()?;
        progress(MetalProgressEvent::prefill_tokens(prefix.len(), prompt.token_ids.len()));
        let logits = step::forward_token(
            &self.model,
            &self.stream,
            &mut state,
            last,
            prefix.len(),
            sampling == SamplingLogits::None,
        )?;
        state.position = prompt.token_ids.len();
        let output = step::output(&self.model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        progress(MetalProgressEvent::prefill_tokens(
            prompt.token_ids.len(),
            prompt.token_ids.len(),
        ));
        Ok(NativePrefill { output, prefix_cache_tokens: 0 })
    }

    pub(super) fn prefill_spatial_merge_vision(
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
        let Some(LoadedVisionModel::SpatialMergeEncoder(tower)) = self.vision_model.as_ref() else {
            return Err(Error::UnsupportedModel(
                "spatial-merge vision tower is not loaded or its tensors are incomplete".into(),
            ));
        };
        let crate::engine::DecoderModel::HybridLinearMoe(decoder) = &self.model else {
            return Err(Error::UnsupportedModel(
                "spatial-merge vision Metal multimodal prefill requires the hybrid linear MoE decoder".into(),
            ));
        };
        let mut state = SessionState::new(decoder.new_cache(&self.stream)?);
        state.rope_position_delta = prompt.position_delta;
        state
            .cache
            .reserve(prompt.token_ids.len().max(self.stream.config().cache.kv_reserve_tokens))?;
        progress(MetalProgressEvent::prefill_tokens(0, prompt.token_ids.len()));
        let prefix_prompt = spatial_merge_prefix(prompt, prefix.len());
        let hidden = tower.forward_multimodal_prefill(
            decoder, &prefix_prompt, image, &mut state.cache, &self.stream,
        )?;
        hidden.async_eval()?;
        self.stream.synchronize()?;
        progress(MetalProgressEvent::prefill_tokens(prefix.len(), prompt.token_ids.len()));
        state.position = prefix.len();
        let model_position = state.model_position()?;
        let logits = step::forward_token(
            &self.model,
            &self.stream,
            &mut state,
            last,
            model_position,
            sampling == SamplingLogits::None,
        )?;
        state.position = prompt.token_ids.len();
        let output = step::output(&self.model, &self.stream, &mut state, logits, sampling)?;
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
