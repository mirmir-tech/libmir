use super::{Error, LoadedModel, Result, Sequence, step};

impl Sequence {
    pub(super) fn complete_terminal_chunk(
        &mut self,
        loaded: &mut LoadedModel,
        remaining: usize,
        count: usize,
        budget: usize,
    ) -> Result<Option<usize>> {
        if count >= budget
            || !crate::native::prefill::tail::can_finish(
                loaded,
                self.position,
                remaining,
                count,
                self.checkpoints.get(self.next_checkpoint).copied(),
            )
        {
            return Ok(None);
        }
        let model = loaded.execution.decoder()?;
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| Error::InvalidPrefillBatch("prefill sequence has no state".into()))?;
        let hidden = step::forward_prefill_state(
            model,
            &loaded.stream,
            state,
            &self.request.prompt_tokens[self.position..],
            self.position,
        )?;
        let logits = model.prefill_logits(&hidden, &loaded.stream)?;
        self.complete(loaded, logits)?;
        Ok(Some(count + 1))
    }
}
