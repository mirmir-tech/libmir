mod completion;
mod retention;

use runtime::backend::{PrefillRequest, SamplingLogits};

use super::{
    super::{NativePrefill, evaluation, timing::PrefillTiming},
    reservation,
};
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
    pub(super) page_reservation_pending: bool,
    pub(super) reservation: reservation::Plan,
    pub output: Option<NativePrefill>,
    pub timing: PrefillTiming,
}

impl Sequence {
    pub(super) fn reclaim_reservation(&mut self) -> Result<()> {
        if let Some(state) = self.state.as_mut() {
            let target =
                self.request.prompt_tokens.len().saturating_add(self.reservation.page_size);
            state.cache.release_reservation_after(target)?;
            self.reservation.tokens = self.reservation.tokens.min(target);
        }
        Ok(())
    }

    pub(super) fn plan_reservation(&mut self) -> Result<()> {
        self.state
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?
            .cache
            .plan_contiguous(self.reservation.tokens());
        Ok(())
    }

    pub fn prepare(
        loaded: &LoadedModel,
        request: PrefillRequest,
        execution_sampling: SamplingLogits,
        restored: Option<RestoredPrefix>,
        timing: PrefillTiming,
    ) -> Result<Self> {
        let model = loaded.execution.decoder()?;
        if request.prompt_tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let (mut state, position, cached_logits) = if let Some((state, logits)) = restored {
            let position = state.position;
            (state, position, logits)
        } else {
            (SessionState::new(model.new_cache(&loaded.stream)?), 0, None)
        };
        let reserve =
            request.prompt_tokens.len().max(loaded.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        let reservation = reservation::Plan::new(
            request.prompt_tokens.len(),
            request.generation_tokens,
            &loaded.stream,
        );
        state.cache.plan_contiguous(reservation.tokens());
        let checkpoints = request
            .cache_checkpoints
            .iter()
            .copied()
            .filter(|checkpoint| {
                *checkpoint > position && *checkpoint < request.prompt_tokens.len()
            })
            .collect();
        Ok(Self {
            request,
            execution_sampling,
            state: Some(state),
            position,
            prefix_cache_tokens: position,
            checkpoints,
            next_checkpoint: 0,
            cached_logits,
            page_reservation_pending: true,
            reservation,
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
        let required = sequences.iter().map(|sequence| reservation::required(sequence)).sum();
        loaded.reserve_prefill_pages(required)?;
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
        evaluation::materialize_packed(loaded, &states, &state_root)?;
        for sequence in sequences {
            sequence.page_reservation_pending = false;
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
            reservation::ensure(self, loaded)?;
            let remaining = prefix_len - self.position;
            let count = loaded
                .prefill_chunk_len(self.position, remaining)
                .min(budget)
                .min(self.checkpoint_distance());
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
            evaluation::materialize(loaded, state, &state_root)?;
            self.page_reservation_pending = false;
            self.position += count;
            self.cache_checkpoint(loaded)?;
            return Ok(count);
        }
        reservation::ensure(self, loaded)?;
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
        self.page_reservation_pending = false;
        self.complete(loaded, logits)?;
        Ok(1)
    }

    pub(super) fn into_state(self) -> Option<SessionState> {
        self.state
    }
}
