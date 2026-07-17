use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{
    error::{Error, Result},
    model::{LoadedModel, NativeOutput},
    session::SessionState,
    step,
};
use crate::MetalProgressEvent;

pub(super) struct NativePrefill {
    pub(super) output: NativeOutput,
    pub(super) prefix_cache_tokens: usize,
}

impl LoadedModel {
    pub(super) fn prefill(
        &mut self,
        session: Uuid,
        tokens: &[u32],
        sampling: SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let Some((&last, prefix)) = tokens.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        let restored = self.prefixes.restore_longest(&self.info.manifest.id, tokens)?;
        let (mut state, mut position, prefix_cache_tokens, cached_logits) =
            if let Some((state, logits)) = restored {
                let position = state.position;
                (state, position, position, Some(logits))
            } else {
                (SessionState::new(self.model.new_cache()?), 0, 0, None)
            };
        let reserve = tokens.len().max(self.stream.config().cache.kv_reserve_tokens);
        state.cache.reserve(reserve)?;
        if position == tokens.len() {
            let logits = cached_logits.ok_or(Error::NoPrefixLogits)?;
            let output = step::output(&self.model, &self.stream, &mut state, logits, sampling)?;
            self.sessions.insert(session, state);
            progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
            return Ok(NativePrefill { output, prefix_cache_tokens });
        }

        progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
        let mut remaining = &prefix[position..];
        while !remaining.is_empty() {
            let count = self.prefill_chunk_len(position, remaining.len());
            let logits = step::forward_prefill(
                &self.model,
                &self.stream,
                &mut state,
                &remaining[..count],
                position,
            )?;
            logits.async_eval()?;
            self.stream.synchronize()?;
            position += count;
            remaining = &remaining[count..];
            progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
        }
        let logits = step::forward_token(
            &self.model,
            &self.stream,
            &mut state,
            last,
            position,
            sampling == SamplingLogits::None,
        )?;
        state.position = tokens.len();
        self.prefixes.insert(&self.info.manifest.id, tokens, &state, &logits)?;
        progress(MetalProgressEvent::prefill_tokens(tokens.len(), tokens.len()));
        let output = step::output(&self.model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        Ok(NativePrefill { output, prefix_cache_tokens })
    }
}
