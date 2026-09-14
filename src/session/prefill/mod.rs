use runtime::backend::{PrefillOutput, PrefillRequest, SamplingLogits};

use super::Session;
use crate::{CancellationToken, ProgressEvent, Result, model::FillClaim};

impl Session {
    /// Prefills this session with a complete prompt and returns the first
    /// prediction.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        self.prefill_reserved(tokens, &[], 0, sampling, &CancellationToken::default(), progress)
    }

    /// Prefills a text prompt, observing cancellation between scheduled
    /// accelerator steps. A cancelled session must be dropped before being
    /// reused.
    pub fn prefill_cancellable(
        &mut self,
        tokens: &[u32],
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(ProgressEvent),
        cancellation: &CancellationToken,
    ) -> Result<PrefillOutput> {
        self.prefill_reserved(tokens, &[], 0, sampling, cancellation, progress)
    }

    pub(crate) fn prefill_generation_reserved(
        &mut self,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        reserved_tokens: usize,
        sampling: SamplingLogits,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        self.prefill_reserved(
            tokens,
            cache_checkpoints,
            reserved_tokens,
            sampling,
            cancellation,
            progress,
        )
    }

    fn prefill_reserved(
        &mut self,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        reserved_tokens: usize,
        sampling: SamplingLogits,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        cancellation.check()?;
        let cache_started = std::time::Instant::now();
        let mut fill_wait = std::time::Duration::ZERO;
        let (admission, counters_before, fill_guard) = loop {
            cancellation.check()?;
            let (admission, counters) = self.model.clone().with_cache(|cache| {
                let admission =
                    self.state.probe_prefill_admission(cache, tokens, reserved_tokens)?;
                Ok((admission, cache.stats().counters))
            })?;
            match self.model.claim_cache_fill(
                tokens,
                cache_checkpoints,
                tokens.len().saturating_sub(admission.missing_tokens),
                cancellation,
            )? {
                FillClaim::Leader(guard) => break (admission, counters, guard),
                FillClaim::Retry(wait) => fill_wait += wait,
            }
        };
        let cohort_wait = self.model.wait_for_cache_cohort(
            admission.needs_eviction,
            admission.missing_tokens,
            cancellation,
        )?;
        let (request, counters_after) =
            self.model.clone().with_cache_wait_cancellable(cancellation, |cache| {
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
            cohort_wait_ms = (cohort_wait + fill_wait).as_secs_f64() * 1_000.0,
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
                generation_tokens: std::num::NonZeroUsize::new(reserved_tokens),
                sampling_logits: sampling,
            },
            reserved_tokens > 1,
            cancellation,
            progress,
        )?;
        self.model
            .clone()
            .with_cache(|cache| Ok(self.state.commit_ready_prefix_blocks(cache)?))?;
        drop(fill_guard);
        output.timings.get_or_insert_default().cache_prepare = cache_prepare;
        Ok(output)
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests;
