use std::time::Instant;

use runtime::backend::{PrefillRequest, SamplingLogits};

use super::super::NativePrefill;
use crate::native::{
    error::{Error, Result},
    model::LoadedModel,
    session::SessionState,
    step,
};

pub(super) struct Sequence {
    pub request: PrefillRequest,
    execution_sampling: SamplingLogits,
    state: Option<SessionState>,
    pub position: usize,
    prefix_cache_tokens: usize,
    cached_logits: Option<crate::engine::Array>,
    pub output: Option<NativePrefill>,
    pub started: Instant,
}

impl Sequence {
    pub fn prepare(
        loaded: &mut LoadedModel,
        request: PrefillRequest,
        execution_sampling: SamplingLogits,
    ) -> Result<Self> {
        let model = loaded.execution.decoder()?;
        if request.prompt_tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let restored = loaded
            .prefixes
            .restore_longest(&loaded.info.manifest.id, &request.prompt_tokens)?;
        let (mut state, position, cached_logits) = if let Some((state, logits)) = restored {
            let position = state.position;
            (state, position, logits)
        } else {
            (SessionState::new(model.new_cache(&loaded.stream)?), 0, None)
        };
        let reserve =
            request.prompt_tokens.len().max(loaded.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        Ok(Self {
            request,
            execution_sampling,
            state: Some(state),
            position,
            prefix_cache_tokens: position,
            cached_logits,
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
            loaded.prefill_chunk_len(self.position, prefix_len - self.position).min(budget)
        })
    }

    pub fn packed_prefill_eligible(&self) -> bool {
        self.prefix_cache_tokens > 0
    }

    pub fn advance_packed(
        loaded: &LoadedModel,
        sequences: &mut [&mut Self],
        count: usize,
    ) -> Result<()> {
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
        state_root.async_eval()?;
        loaded.stream.synchronize()?;
        let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
        for sequence in sequences {
            sequence.position += count;
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
            let remaining = prefix_len - self.position;
            let count = loaded.prefill_chunk_len(self.position, remaining).min(budget);
            let tokens = &self.request.prompt_tokens[self.position..self.position + count];
            let model = loaded.execution.decoder()?;
            let state = self.state.as_mut().ok_or_else(|| {
                Error::InvalidPrefillBatch("prefill sequence has no state".into())
            })?;
            let state_root =
                step::forward_prefill_state(model, &loaded.stream, state, tokens, self.position)?;
            state_root.async_eval()?;
            loaded.stream.synchronize()?;
            let _reclaimed = LoadedModel::reclaim_prefill_allocator_cache()?;
            self.position += count;
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
