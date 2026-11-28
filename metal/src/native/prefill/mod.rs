use runtime::{
    backend::SamplingLogits,
    kv::{BlockTable, reusable_prefix_tokens},
};
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
pub(in crate::native) mod evaluation;
mod scalar;
#[cfg(test)]
mod tail;
pub(in crate::native) mod timing;
pub(in crate::native) mod validation;
mod vision;

pub use batch::MetalPrefillBatch;
pub(in crate::native) use batch::{ActivePrefills, PrefillStep, RetirePrefill};
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
        sampling: &SamplingLogits,
        prefix_block_size: Option<usize>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let block_table =
            prefix_block_size.map_or_else(BlockTable::default, BlockTable::with_block_size);
        self.prefill_with_budget(
            session,
            tokens,
            cache_checkpoints,
            sampling,
            &block_table,
            0,
            progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_with_budget(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        sampling: &SamplingLogits,
        block_table: &BlockTable,
        cached_tokens: usize,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        self.require_execution_ready()?;
        if tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let (mut state, cached_logits) =
            self.prepare_prefill_state(tokens, block_table, cached_tokens)?;
        let position = state.position;
        let prefix_cache_tokens = position;
        if position == 0 && self.has_non_paged_state() {
            self.prefixes.reserve_miss_slot();
        }
        let result = (|| {
            let model = self.execution.decoder()?;
            if position == tokens.len() {
                let logits = cached_logits.ok_or(Error::NoPrefixLogits)?;
                let output = diagnostics::measure(diagnostics::Stage::FirstToken, || {
                    step::output(model, &self.stream, &mut state, logits, sampling)
                })?;
                progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
                return Ok(NativePrefill { output, prefix_cache_tokens });
            }

            progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
            let logits = self.forward_prefill_prompt(
                &mut state,
                tokens,
                cache_checkpoints,
                position,
                sampling,
                progress,
            )?;
            evaluation::materialize(self, &state, &logits)?;
            state.position = tokens.len();
            if self.has_non_paged_state() {
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
                    block_table.block_size(),
                    prefix_bytes,
                )?;
            }
            let _reclaimed = Self::reclaim_prefill_allocator_cache()?;
            progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
            let model = self.execution.decoder()?;
            let output = diagnostics::measure(diagnostics::Stage::FirstToken, || {
                step::output(model, &self.stream, &mut state, logits, sampling)
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

    fn prepare_prefill_state(
        &mut self,
        tokens: &[u32],
        block_table: &BlockTable,
        cached_tokens: usize,
    ) -> Result<super::prefix::RestoredPrefix> {
        self.apply_history_pressure(crate::engine::memory_stats()?, tokens.len())?;
        let blocks = block_table.blocks();
        if !self.has_non_paged_state() {
            let reusable = if blocks.is_empty() {
                0
            } else {
                reusable_prefix_tokens(cached_tokens, tokens.len(), block_table.block_size(), 0)
            };
            let model = self.execution.decoder()?;
            let mut state = SessionState::from_prefix(model.new_cache(&self.stream)?, reusable);
            if !blocks.is_empty() {
                state.cache.bind_blocks(blocks, Some(reusable), &self.stream)?;
            }
            return Ok((state, None));
        }
        let restored = self
            .prefixes
            .lease_longest(&self.info.manifest.id, tokens)?
            .map(|lease| lease.restored)
            .filter(|(state, _)| blocks.is_empty() || state.position <= cached_tokens);
        let (mut state, logits) = if let Some(restored) = restored {
            restored
        } else {
            let model = self.execution.decoder()?;
            (SessionState::new(model.new_cache(&self.stream)?), None)
        };
        if !blocks.is_empty() {
            state.cache.bind_blocks(blocks, None, &self.stream)?;
        }
        Ok((state, logits))
    }
}
