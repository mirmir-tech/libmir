mod completion;
mod retention;
#[cfg(test)]
mod terminal;

use runtime::backend::{PrefillRequest, SamplingLogits};

use super::super::{NativePrefill, evaluation, timing::PrefillTiming};
use crate::native::{
    error::{Error, Result},
    model::LoadedModel,
    prefix::RestoredPrefix,
    session::SessionState,
    step,
};

pub(super) struct Sequence {
    pub request: PrefillRequest,
    execution_sampling: SamplingLogits,
    state: Option<SessionState>,
    pub position: usize,
    prefix_cache_tokens: usize,
    checkpoints: Vec<usize>,
    next_checkpoint: usize,
    cached_logits: Option<crate::engine::Array>,
    pub output: Option<NativePrefill>,
    pub timing: PrefillTiming,
}

impl Sequence {
    pub fn prepare(
        loaded: &LoadedModel,
        request: PrefillRequest,
        execution_sampling: SamplingLogits,
        restored: Option<RestoredPrefix>,
        timing: PrefillTiming,
    ) -> Result<Self> {
        if request.prompt_tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let (state, cached_logits) = super::super::resume(
            loaded,
            &request.block_table,
            request.cached_tokens,
            request.prompt_tokens.len(),
            restored,
        )?;
        let position = state.position;
        let checkpoints = if loaded.has_non_paged_state() {
            request
                .cache_checkpoints
                .iter()
                .copied()
                .filter(|checkpoint| {
                    *checkpoint > position && *checkpoint < request.prompt_tokens.len()
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            request,
            execution_sampling,
            state: Some(state),
            position,
            prefix_cache_tokens: position,
            checkpoints,
            next_checkpoint: 0,
            cached_logits,
            output: None,
            timing,
        })
    }

    pub fn prefill_count(&self, loaded: &LoadedModel, budget: usize) -> Option<usize> {
        let prefix_len = self.request.prompt_tokens.len().saturating_sub(1);
        (self.position < prefix_len).then(|| {
            loaded
                .prefill_chunk_len(self.position, prefix_len - self.position)
                .min(budget)
                .min(self.checkpoint_distance())
        })
    }

    #[tracing::instrument(target = "libmir::metal::prefill", level = "debug", skip_all,
        fields(rows = sequences.len(), position = sequences[0].position, count = count))]
    pub fn advance_packed(
        loaded: &mut LoadedModel,
        sequences: &mut [&mut Self],
        count: usize,
    ) -> Result<()> {
        let positions = sequences.iter().map(|sequence| sequence.position).collect::<Vec<_>>();
        #[cfg(test)]
        let checkpoint_rows = sequences
            .iter()
            .enumerate()
            .filter(|(_, sequence)| {
                loaded.prefixes.enabled() && sequence.retains_checkpoint_after(count)
            })
            .map(|(row, _)| row)
            .collect::<Vec<_>>();
        let tokens = sequences
            .iter()
            .flat_map(|sequence| {
                sequence.request.prompt_tokens[sequence.position..sequence.position + count]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        let mut states = sequences
            .iter_mut()
            .map(|sequence| {
                sequence.state.as_mut().ok_or_else(|| {
                    Error::InvalidPrefillBatch("prefill sequence has no state".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let model = loaded.execution.decoder()?;
        let state_root = step::forward_packed_prefill_state(
            model, &loaded.stream, &mut states, &positions, &tokens, count,
        )?;
        #[cfg(test)]
        for row in checkpoint_rows {
            states[row].cache.prepare_prefix_retention(&loaded.stream)?;
        }
        evaluation::intermediate_packed(loaded, &states, state_root)?;
        for sequence in sequences {
            sequence.position += count;
            sequence.cache_checkpoint(loaded)?;
        }
        Ok(())
    }

    #[tracing::instrument(target = "libmir::metal::prefill", level = "debug", skip_all,
        fields(rows = 1, position = self.position, budget = budget))]
    pub fn advance(&mut self, loaded: &mut LoadedModel, budget: usize) -> Result<usize> {
        let prompt_len = self.request.prompt_tokens.len();
        if self.position == prompt_len {
            let logits = self.cached_logits.take().ok_or(Error::NoPrefixLogits)?;
            self.complete(loaded, logits)?;
            return Ok(1);
        }
        let prefix_len = prompt_len - 1;
        if self.position < prefix_len {
            let remaining = prefix_len - self.position;
            let count = loaded
                .prefill_chunk_len(self.position, remaining)
                .min(budget)
                .min(self.checkpoint_distance());
            #[cfg(test)]
            if let Some(used) = self.complete_terminal_chunk(loaded, remaining, count, budget)? {
                return Ok(used);
            }
            let tokens = &self.request.prompt_tokens[self.position..self.position + count];
            #[cfg(test)]
            let checkpoint = loaded.prefixes.enabled() && self.retains_checkpoint_after(count);
            let model = loaded.execution.decoder()?;
            let state = self.state.as_mut().ok_or_else(|| {
                Error::InvalidPrefillBatch("prefill sequence has no state".into())
            })?;
            let state_root =
                step::forward_prefill_state(model, &loaded.stream, state, tokens, self.position)?;
            #[cfg(test)]
            if checkpoint {
                state.cache.prepare_prefix_retention(&loaded.stream)?;
            }
            evaluation::intermediate(loaded, state, state_root)?;
            self.position += count;
            self.cache_checkpoint(loaded)?;
            return Ok(count);
        }
        let model = loaded.execution.decoder()?;
        let last = self.request.prompt_tokens[prefix_len];
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?;
        let logits = step::forward_token(
            model,
            &loaded.stream,
            state,
            last,
            self.position,
            self.execution_sampling == SamplingLogits::None,
        )?;
        self.complete(loaded, logits)?;
        Ok(1)
    }

    pub(super) fn into_state(self) -> Option<SessionState> {
        self.state
    }
}
