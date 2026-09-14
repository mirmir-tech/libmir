use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{
    error::{Error, Result},
    model::{LoadedModel, NativeOutput},
    session::SessionState,
    step,
};
use crate::MetalProgressEvent;

mod batch;
mod cohort;
pub(in crate::native) mod diagnostics;
mod evaluation;
mod reservation;
#[cfg(test)]
mod tests;
pub(in crate::native) mod timing;
pub(in crate::native) mod validation;
mod vision;

pub use batch::MetalPrefillBatch;
pub(in crate::native) use batch::{PrefillStep, Reservations, RetirePrefill};
pub use cohort::MetalPrefillCohort;

pub(super) struct NativePrefill {
    pub(super) output: NativeOutput,
    pub(super) prefix_cache_tokens: usize,
}

impl LoadedModel {
    #[cfg(test)]
    pub(super) fn prefill(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        sampling: SamplingLogits,
        prefix_block_size: Option<usize>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        self.prefill_with_budget(
            session,
            tokens,
            cache_checkpoints,
            sampling,
            prefix_block_size,
            None,
            progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_with_budget(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        sampling: SamplingLogits,
        prefix_block_size: Option<usize>,
        generation_tokens: Option<std::num::NonZeroUsize>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        self.require_execution_ready()?;
        let Some((&last, prefix)) = tokens.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        let (mut state, cached_logits) = self.prepare_prefill_state(tokens)?;
        let mut position = state.position;
        let prefix_cache_tokens = position;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        let plan = reservation::Plan::admit(self, tokens.len(), generation_tokens, position)?;
        state.cache.plan_contiguous(plan.tokens());
        if position == 0 {
            self.prefixes.reserve_miss_slot();
        }
        let result = (|| {
            let model = self.execution.decoder()?;
            if position == tokens.len() {
                let logits = cached_logits.ok_or(Error::NoPrefixLogits)?;
                let output = diagnostics::measure(diagnostics::Stage::FirstToken, || {
                    reservation::output(model, &self.stream, &mut state, logits, sampling)
                })?;
                progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
                return Ok(NativePrefill { output, prefix_cache_tokens });
            }

            progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
            let mut remaining = &prefix[position..];
            let restored_position = position;
            let mut checkpoints = cache_checkpoints
                .iter()
                .copied()
                .filter(|checkpoint| *checkpoint > restored_position && *checkpoint < tokens.len())
                .peekable();
            while !remaining.is_empty() {
                let count = self
                    .prefill_chunk_len(position, remaining.len())
                    .min(checkpoints.peek().map_or(usize::MAX, |checkpoint| checkpoint - position));
                let state_root = step::forward_prefill_state(
                    model,
                    &self.stream,
                    &mut state,
                    &remaining[..count],
                    position,
                )?;
                #[cfg(test)]
                if self.prefixes.enabled() && checkpoints.peek() == Some(&(position + count)) {
                    state.cache.prepare_prefix_retention(&self.stream)?;
                }
                evaluation::materialize(self, &state, &state_root)?;
                position += count;
                remaining = &remaining[count..];
                if checkpoints.next_if_eq(&position).is_some() {
                    let checkpoint_bytes = self.estimated_prefix_bytes(position)?;
                    super::model::cache_prefix_checkpoint(
                        &mut self.prefixes,
                        &self.info.manifest.id,
                        &tokens[..position],
                        &state,
                        page_size,
                        checkpoint_bytes,
                    )?;
                }
                progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
            }
            let logits = step::forward_token(
                model,
                &self.stream,
                &mut state,
                last,
                position,
                sampling == SamplingLogits::None,
            )?;
            evaluation::materialize(self, &state, &logits)?;
            state.position = tokens.len();
            let prefix_bytes = self
                .estimated_prefix_bytes(tokens.len())?
                .checked_add(logits.byte_len()?)
                .ok_or(crate::engine::Error::ShapeOverflow)?;
            let _cached = super::model::cache_prefix_snapshot(
                &mut self.prefixes,
                &self.info.manifest.id,
                tokens,
                &state,
                &logits,
                prefix_block_size,
                prefix_bytes,
            )?;
            let _reclaimed = Self::reclaim_prefill_allocator_cache()?;
            progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
            let output = diagnostics::measure(diagnostics::Stage::FirstToken, || {
                reservation::output(model, &self.stream, &mut state, logits, sampling)
            })?;
            Ok(NativePrefill { output, prefix_cache_tokens })
        })();
        match result {
            Ok(output) => {
                self.sessions.insert(session, state);
                Ok(output)
            },
            Err(error) => self.fail_execution([session], [state], error),
        }
    }

    fn prepare_prefill_state(&mut self, tokens: &[u32]) -> Result<super::prefix::RestoredPrefix> {
        #[cfg(test)]
        self.apply_history_pressure(crate::engine::memory_stats()?)?;
        let restored = self
            .prefixes
            .lease_longest(&self.info.manifest.id, tokens)?
            .map(|lease| lease.restored);
        let (mut state, logits) = if let Some(restored) = restored {
            restored
        } else {
            let model = self.execution.decoder()?;
            (SessionState::new(model.new_cache(&self.stream)?), None)
        };
        state
            .cache
            .reserve(tokens.len().max(self.stream.config().cache.kv_reserve_tokens))?;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        state.cache.plan_contiguous(tokens.len().saturating_add(page_size));
        Ok((state, logits))
    }
}

fn required_prefill_pages(tokens: usize, position: usize, page_size: usize) -> usize {
    let page_size = page_size.max(1);
    let planned = tokens.saturating_add(page_size).div_ceil(page_size);
    let owned = position.div_ceil(page_size);
    let copy_on_write = usize::from(position > 0 && !position.is_multiple_of(page_size));
    planned.saturating_sub(owned).saturating_add(copy_on_write)
}
