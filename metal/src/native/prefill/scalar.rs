use runtime::backend::SamplingLogits;

use super::{LoadedModel, evaluation};
use crate::{
    MetalProgressEvent,
    engine::Array,
    native::{error::Result, session::SessionState, step},
};

impl LoadedModel {
    pub(super) fn forward_prefill_prompt(
        &mut self,
        state: &mut SessionState,
        tokens: &[u32],
        cache_checkpoints: &[usize],
        mut position: usize,
        sampling: &SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Array> {
        let model = self.execution.decoder()?;
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        let prefix = &tokens[..tokens.len() - 1];
        let last = tokens[tokens.len() - 1];
        let mut remaining = &prefix[position..];
        #[cfg(not(test))]
        let terminal_logits = None;
        #[cfg(test)]
        let mut terminal_logits = None;
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
            #[cfg(test)]
            if super::tail::can_finish(
                self,
                position,
                remaining.len(),
                count,
                checkpoints.peek().copied(),
            ) {
                let hidden = step::forward_prefill_state(
                    model,
                    &self.stream,
                    state,
                    &tokens[position..],
                    position,
                )?;
                terminal_logits = Some(model.prefill_logits(&hidden, &self.stream)?);
                break;
            }
            let state_root = step::forward_prefill_state(
                model,
                &self.stream,
                state,
                &remaining[..count],
                position,
            )?;
            #[cfg(test)]
            if self.prefixes.enabled() && checkpoints.peek() == Some(&(position + count)) {
                state.cache.prepare_prefix_retention(&self.stream)?;
            }
            #[cfg(not(test))]
            let deferred = false;
            #[cfg(test)]
            let deferred = super::tail::can_defer(
                self,
                position,
                remaining.len(),
                count,
                checkpoints.peek().copied(),
            );
            if !deferred {
                evaluation::intermediate(self, state, state_root)?;
            }
            position += count;
            remaining = &remaining[count..];
            if checkpoints.next_if_eq(&position).is_some() {
                let checkpoint_bytes = self.estimated_prefix_bytes(position)?;
                crate::native::model::cache_prefix_checkpoint(
                    &mut self.prefixes,
                    &self.info.manifest.id,
                    &tokens[..position],
                    state,
                    page_size,
                    checkpoint_bytes,
                )?;
            }
            if !deferred {
                progress(MetalProgressEvent::prefill_tokens(position, tokens.len()));
            }
        }
        let logits = terminal_logits.map_or_else(
            || {
                step::forward_token(
                    model,
                    &self.stream,
                    state,
                    last,
                    position,
                    *sampling == SamplingLogits::None,
                )
            },
            Ok,
        )?;
        Ok(logits)
    }
}
