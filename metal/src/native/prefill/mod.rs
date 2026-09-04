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
mod evaluation;
#[cfg(test)]
mod tests;
mod vision;

pub use batch::MetalPrefillBatch;
pub(in crate::native) use batch::PrefillStep;
pub use cohort::MetalPrefillCohort;

pub(super) struct NativePrefill {
    pub(super) output: NativeOutput,
    pub(super) prefix_cache_tokens: usize,
}

impl LoadedModel {
    pub(super) fn prefill(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        sampling: SamplingLogits,
        prefix_block_size: Option<usize>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let Some((&last, prefix)) = tokens.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        let restored = self.prefixes.restore_longest(&self.info.manifest.id, tokens)?;
        let (mut state, mut position, prefix_cache_tokens, cached_logits) =
            if let Some((state, logits)) = restored {
                let position = state.position;
                (state, position, position, logits)
            } else {
                let model = self.execution.decoder()?;
                (SessionState::new(model.new_cache(&self.stream)?), 0, 0, None)
            };
        let reserve = tokens.len().max(self.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        state.cache.plan_contiguous(tokens.len().saturating_add(page_size));
        let required_pages = required_prefill_pages(tokens.len(), position, page_size);
        self.reserve_prefill_pages(required_pages)?;
        let model = self.execution.decoder()?;
        if position == tokens.len() {
            let logits = cached_logits.ok_or(Error::NoPrefixLogits)?;
            let output = step::output(model, &self.stream, &mut state, logits, sampling)?;
            self.sessions.insert(session, state);
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
        let output = step::output(model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        Ok(NativePrefill { output, prefix_cache_tokens })
    }
}

fn required_prefill_pages(tokens: usize, position: usize, page_size: usize) -> usize {
    let page_size = page_size.max(1);
    let planned = tokens.saturating_add(page_size).div_ceil(page_size);
    let owned = position.div_ceil(page_size);
    let copy_on_write = usize::from(position > 0 && !position.is_multiple_of(page_size));
    planned.saturating_sub(owned).saturating_add(copy_on_write)
}
