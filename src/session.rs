use runtime::{
    backend::{DecodeOutput, DecodeSequence, PrefillOutput, PrefillRequest, SamplingLogits},
    kv::KvSessionState,
};
use uuid::Uuid;

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::PreparedVisionPrompt;
use crate::{Model, ProgressEvent, Result, scheduler::PendingModelDecode};

/// Stateful low-level inference session with independent accelerator K/V state.
pub struct Session {
    model: Model,
    state: KvSessionState,
}

pub struct PendingSessionDecode {
    pending: PendingModelDecode,
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
        self.prefill_reserved(tokens, &[], 0, sampling, false, progress)
    }

    pub(crate) fn prefill_generation_reserved(
        &mut self,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        reserved_tokens: usize,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        self.prefill_reserved(
            tokens,
            cache_checkpoints,
            reserved_tokens,
            sampling,
            reserved_tokens > 1,
            progress,
        )
    }

    fn prefill_reserved(
        &mut self,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        reserved_tokens: usize,
        sampling: SamplingLogits,
        expects_decode: bool,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let cache_started = std::time::Instant::now();
        let (admission, counters_before) = self.model.clone().with_cache(|cache| {
            let admission = self.state.probe_prefill_admission(cache, tokens, reserved_tokens)?;
            Ok((admission, cache.stats().counters))
        })?;
        let cohort_wait = self
            .model
            .wait_for_cache_cohort(admission.needs_eviction, admission.missing_tokens);
        let (request, counters_after) = self.model.clone().with_cache_wait(|cache| {
            let request = self
                .state
                .prepare_prefill_with_reserve_in_place(cache, tokens, reserved_tokens)?;
            Ok((request, cache.stats().counters))
        })?;
        tracing::debug!(
            session = %request.session_id,
            cached_tokens = request.cached_tokens,
            missing_tokens = request.missing_tokens,
            reserved_tokens,
            needs_eviction = admission.needs_eviction,
            cohort_wait_ms = cohort_wait.as_secs_f64() * 1_000.0,
            cache_evictions = counters_after.evictions,
            cache_protected_prefix_skips = counters_after.protected_prefix_skips,
            evictions_since_probe = counters_after.evictions.saturating_sub(counters_before.evictions),
            protected_skips_since_probe = counters_after
                .protected_prefix_skips
                .saturating_sub(counters_before.protected_prefix_skips),
            "prepared cache-aware prefill allocation"
        );
        let cache_prepare = cache_started.elapsed();
        let mut output = self.model.prefill_request(
            PrefillRequest {
                model: self.model.handle().clone(),
                session_id: request.session_id,
                prompt_tokens: tokens.to_vec(),
                cache_checkpoints: cache_checkpoints.to_vec(),
                block_table: self.state.table().clone(),
                cached_tokens: request.cached_tokens,
                sampling_logits: sampling,
            },
            expects_decode,
            progress,
        )?;
        self.model
            .clone()
            .with_cache(|cache| Ok(self.state.commit_ready_prefix_blocks(cache)?))?;
        output.timings.get_or_insert_default().cache_prepare = cache_prepare;
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
        self.prefill_vision_reserved(prepared, 0, sampling, progress)
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(crate) fn prefill_vision_reserved(
        &mut self,
        prepared: &PreparedVisionPrompt,
        reserved_tokens: usize,
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        let tokens = match prepared {
            PreparedVisionPrompt::Pooled { tokens, .. } => &tokens.token_ids,
            PreparedVisionPrompt::SpatialMerge { tokens, .. } => &tokens.token_ids,
        };
        let request = self.model.clone().with_cache_wait(|cache| {
            Ok(self
                .state
                .prepare_uncached_prefill_with_reserve_in_place(cache, tokens, reserved_tokens)?)
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
        let pending = self.start_decode(token, sampling)?;
        self.finish_decode(pending)
    }

    pub(crate) fn start_decode(
        &mut self,
        token: u32,
        sampling: SamplingLogits,
    ) -> Result<PendingSessionDecode> {
        let request = self
            .model
            .clone()
            .with_cache_wait(|cache| Ok(self.state.append_decode_in_place(cache, token)?))?;
        let pending = self.model.start_decode_sequence(DecodeSequence {
            session_id: request.session_id,
            token_id: token,
            block_table: self.state.table().clone(),
            sampling_logits: sampling,
        })?;
        Ok(PendingSessionDecode { pending })
    }

    pub(crate) fn finish_decode(&mut self, pending: PendingSessionDecode) -> Result<DecodeOutput> {
        let output = self.model.finish_decode_sequence(pending.pending)?;
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
        self.model.release_decode_session(self.state.session_id());
        let _backend_release = self
            .model
            .engine()
            .release_session(self.model.handle(), self.state.session_id());
        let _cache_release =
            self.model.clone().with_cache(|cache| Ok(self.state.release(cache)?));
        self.model.notify_cache_waiters();
    }
}
