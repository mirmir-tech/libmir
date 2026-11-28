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
            self.release_outgrown_arenas()?;
            state.position = tokens.len();
            let prefix_bytes = self
                .prefix_snapshot_bytes(tokens.len())?
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
                &self.stream,
            )?;
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
        let restored = self
            .prefixes
            .lease_longest_within(
                &self.info.manifest.id,
                tokens,
                (!block_table.blocks().is_empty()).then_some(cached_tokens),
            )?
            .map(|lease| lease.restored);
        resume(self, block_table, cached_tokens, tokens.len(), restored)
    }
}

/// The state a prompt of `prompt_len` tokens resumes from, bound to its
/// blocks. A restored snapshot writes its partial last pages into them.
/// Without one, a model with unpaged state starts over, and a paged model
/// resumes from the runtime's whole cached blocks, recomputing the last
/// token for its logits; so does a paged model whose snapshot resumes no
/// further.
pub(in crate::native) fn resume(
    loaded: &LoadedModel,
    block_table: &BlockTable,
    cached_tokens: usize,
    prompt_len: usize,
    restored: Option<super::prefix::RestoredPrefix>,
) -> Result<super::prefix::RestoredPrefix> {
    let blocks = block_table.blocks();
    let reusable = if blocks.is_empty() || loaded.has_non_paged_state() {
        0
    } else {
        reusable_prefix_tokens(cached_tokens, prompt_len, block_table.block_size(), 0)
    };
    let restored = restored.filter(|(state, logits)| {
        loaded.has_non_paged_state() || logits.is_some() || state.position > reusable
    });
    let Some((mut state, logits)) = restored else {
        let model = loaded.execution.decoder()?;
        let mut state = SessionState::from_prefix(model.new_cache(&loaded.stream)?, reusable);
        if !blocks.is_empty() {
            state.cache.bind_blocks(blocks, Some(reusable), &loaded.stream)?;
        }
        return Ok((state, None));
    };
    if !blocks.is_empty() {
        state.cache.bind_blocks(blocks, None, &loaded.stream)?;
        state.cache.restore_tails(&loaded.stream)?;
    }
    Ok((state, logits))
}
