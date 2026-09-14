use super::Sequence;
use crate::native::{
    error::{Error, Result},
    model::LoadedModel,
};

impl Sequence {
    #[cfg(test)]
    pub(super) fn retains_checkpoint_after(&self, tokens: usize) -> bool {
        self.checkpoints
            .get(self.next_checkpoint)
            .is_some_and(|checkpoint| self.position.checked_add(tokens) == Some(*checkpoint))
    }

    pub(super) fn checkpoint_distance(&self) -> usize {
        self.checkpoints
            .get(self.next_checkpoint)
            .map_or(usize::MAX, |checkpoint| checkpoint.saturating_sub(self.position).max(1))
    }

    pub(super) fn cache_checkpoint(&mut self, loaded: &mut LoadedModel) -> Result<()> {
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
}
