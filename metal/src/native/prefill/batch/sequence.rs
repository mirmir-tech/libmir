use std::time::Instant;

use runtime::backend::{PrefillRequest, SamplingLogits};

use super::{super::NativePrefill, reservation};
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
    pub output: Option<NativePrefill>,
    pub started: Instant,
}

impl Sequence {
    pub fn prepare(
        loaded: &LoadedModel,
        request: PrefillRequest,
        execution_sampling: SamplingLogits,
        restored: Option<RestoredPrefix>,
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
        let page_size = loaded.stream.config().kv_cache.block_size.max(1);
        state
            .cache
            .plan_contiguous(request.prompt_tokens.len().saturating_add(page_size));
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
            output: None,
            started: Instant::now(),
        })
    }

    pub fn pending(&self) -> bool {
        self.output.is_none()
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

    pub fn packed_prefill_eligible(&self) -> bool {
        self.prefix_cache_tokens > 0
    }

    pub fn advance_packed(
        loaded: &mut LoadedModel,
        sequences: &mut [&mut Self],
        count: usize,
    ) -> Result<()> {
        let page_size = loaded.stream.config().kv_cache.block_size.max(1);
        let required = sequences
            .iter()
            .map(|sequence| reservation::required(sequence, page_size))
            .sum();
        loaded.reserve_prefill_pages(required)?;
        let positions = sequences.iter().map(|sequence| sequence.position).collect::<Vec<_>>();
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
        state_root.async_eval(&loaded.stream)?;
        loaded.settle_prefill_graph()?;
        for state in &states {
            state.cache.detach_evaluated_graphs(&loaded.stream)?;
        }
        for sequence in sequences {
            sequence.page_reservation_pending = false;
            sequence.position += count;
            sequence.cache_checkpoint(loaded)?;
        }
        Ok(())
    }

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
            let model = loaded.execution.decoder()?;
            let state = self.state.as_mut().ok_or_else(|| {
                Error::InvalidPrefillBatch("prefill sequence has no state".into())
            })?;
            let state_root =
                step::forward_prefill_state(model, &loaded.stream, state, tokens, self.position)?;
            state_root.async_eval(&loaded.stream)?;
            loaded.settle_prefill_graph()?;
            state.cache.detach_evaluated_graphs(&loaded.stream)?;
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

    fn checkpoint_distance(&self) -> usize {
        self.checkpoints
            .get(self.next_checkpoint)
            .map_or(usize::MAX, |checkpoint| checkpoint.saturating_sub(self.position).max(1))
    }

    fn cache_checkpoint(&mut self, loaded: &mut LoadedModel) -> Result<()> {
        if self.checkpoints.get(self.next_checkpoint) != Some(&self.position) {
            return Ok(());
        }
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?;
        let bytes = loaded.estimated_prefix_bytes(self.position)?;
        crate::native::model::cache_prefix_checkpoint(
            &mut loaded.prefixes,
            &loaded.info.manifest.id,
            &self.request.prompt_tokens[..self.position],
            state,
            loaded.stream.config().kv_cache.block_size.max(1),
            bytes,
        )?;
        self.next_checkpoint += 1;
        Ok(())
    }

    fn complete(&mut self, loaded: &mut LoadedModel, logits: crate::engine::Array) -> Result<()> {
        let tokens = &self.request.prompt_tokens;
        let mut state = self
            .state
            .take()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?;
        state.position = tokens.len();
        let prefix_bytes = loaded
            .estimated_prefix_bytes(tokens.len())?
            .checked_add(logits.byte_len()?)
            .ok_or(crate::engine::Error::ShapeOverflow)?;
        let _cached = crate::native::model::cache_prefix_snapshot(
            &mut loaded.prefixes,
            &loaded.info.manifest.id,
            tokens,
            &state,
            &logits,
            self.request.block_table.block_size(),
            prefix_bytes,
        )?;
        let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
        let model = loaded.execution.decoder()?;
        let output =
            step::output(model, &loaded.stream, &mut state, logits, self.execution_sampling)?;
        loaded.sessions.insert(self.request.session_id, state);
        self.position = tokens.len();
        self.output = Some(NativePrefill {
            output,
            prefix_cache_tokens: self.prefix_cache_tokens,
        });
        Ok(())
    }
}
