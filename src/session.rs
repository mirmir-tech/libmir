use runtime::{
    backend::{DecodeOutput, DecodeSequence, PrefillOutput, SamplingLogits},
    kv::KvSessionState,
};
use uuid::Uuid;

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::PreparedVisionPrompt;
use crate::{Model, ProgressEvent, Result};

/// Stateful low-level inference session with independent accelerator K/V state.
pub struct Session {
    model: Model,
    state: KvSessionState,
}

impl Session {
    pub(super) fn new(model: Model, block_size: usize) -> Self {
        let state = KvSessionState::new(Uuid::new_v4(), &model.handle().id, block_size);
        Self { model, state }
    }

    /// Prefills this session with a complete prompt and returns the first
    /// prediction.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let request = self
            .model
            .clone()
            .with_cache(|cache| Ok(self.state.prepare_prefill_in_place(cache, tokens)?))?;
        let output = self.model.engine().prefill_tokens_with_progress(
            self.model.handle(),
            request.session_id,
            tokens,
            self.state.table(),
            sampling,
            progress,
        )?;
        self.model
            .clone()
            .with_cache(|cache| Ok(self.state.commit_ready_prefix_blocks(cache)?))?;
        Ok(output)
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    /// Prefills a prepared image prompt on the selected accelerator. Multimodal
    /// sessions deliberately never publish reusable prefix-cache entries.
    pub fn prefill_vision(
        &mut self,
        prepared: &PreparedVisionPrompt,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let tokens = match prepared {
            PreparedVisionPrompt::Pooled { tokens, .. } => &tokens.token_ids,
            PreparedVisionPrompt::SpatialMerge { tokens, .. } => &tokens.token_ids,
        };
        let request = self.model.clone().with_cache(|cache| {
            Ok(self.state.prepare_uncached_prefill_in_place(cache, tokens)?)
        })?;
        match prepared {
            PreparedVisionPrompt::Pooled { tokens, image, .. } => {
                Ok(self.model.engine().prefill_pooled_vision_with_progress(
                    self.model.handle(),
                    request.session_id,
                    tokens,
                    image,
                    self.state.table(),
                    sampling,
                    progress,
                )?)
            },
            PreparedVisionPrompt::SpatialMerge { tokens, image, .. } => {
                Ok(self.model.engine().prefill_spatial_merge_vision_with_progress(
                    self.model.handle(),
                    request.session_id,
                    tokens,
                    image,
                    self.state.table(),
                    sampling,
                    progress,
                )?)
            },
        }
    }

    /// Appends one generated token and computes the following prediction.
    pub fn decode(&mut self, token: u32, sampling: SamplingLogits) -> Result<DecodeOutput> {
        let request = self
            .model
            .clone()
            .with_cache(|cache| Ok(self.state.append_decode_in_place(cache, token)?))?;
        let output = self.model.decode_sequence(DecodeSequence {
            session_id: request.session_id,
            token_id: token,
            block_table: self.state.table().clone(),
            sampling_logits: sampling,
        })?;
        self.model
            .clone()
            .with_cache(|cache| Ok(self.state.commit_ready_prefix_blocks(cache)?))?;
        Ok(output)
    }

    #[must_use]
    /// Returns cache statistics shared by sessions of the same loaded model.
    pub fn cache_stats(&self) -> runtime::kv::CacheStats {
        self.model.cache_stats()
    }

    #[must_use]
    /// Returns the loaded model that owns this session.
    pub fn model(&self) -> &Model {
        &self.model
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _backend_release = self
            .model
            .engine()
            .release_session(self.model.handle(), self.state.session_id());
        let _cache_release =
            self.model.clone().with_cache(|cache| Ok(self.state.release(cache)?));
    }
}
