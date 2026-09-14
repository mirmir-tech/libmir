use super::{Arc, Batch, LoadedModel, Mutex, Result};

pub(in crate::native) type RetirePrefill = Box<dyn FnOnce(Batch) + Send + Sync>;

pub(super) struct Cleanup {
    pub(super) inner: Arc<Mutex<Option<Batch>>>,
    pub(super) retire: Option<RetirePrefill>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        let Some(retire) = self.retire.take() else {
            return;
        };
        let batch = match self.inner.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(batch) = batch {
            retire(batch);
        }
    }
}

impl Batch {
    pub(in crate::native) fn cancel(self, loaded: &mut LoadedModel) -> Result<()> {
        if self.sequences.is_empty() {
            return Ok(());
        }
        let sessions = self
            .sequences
            .iter()
            .map(|sequence| sequence.request.session_id)
            .collect::<Vec<_>>();
        let states = self.sequences.into_iter().filter_map(super::Sequence::into_state);
        loaded.retire_execution(sessions, states)
    }
}

impl super::MetalPrefillBatch {
    pub(in crate::native) fn cancel_sessions(
        &self,
        loaded: &mut LoadedModel,
        sessions: &[uuid::Uuid],
    ) -> Result<()> {
        let mut guard = self.inner.lock()?;
        let batch = guard.as_mut().ok_or_else(|| {
            super::Error::InvalidPrefillBatch("prefill batch is already consumed".into())
        })?;
        let removed = batch
            .sequences
            .extract_if(.., |row| sessions.contains(&row.request.session_id))
            .collect::<Vec<_>>();
        batch.cursor %= batch.sequences.len().max(1);
        drop(guard);
        if removed.is_empty() {
            return Ok(());
        }
        // Completed rows live in loaded.sessions; unfinished rows still own their
        // cache here. Keep both until submitted writes have drained.
        let ids = removed.iter().map(|row| row.request.session_id).collect::<Vec<_>>();
        let states = removed.into_iter().filter_map(super::Sequence::into_state);
        loaded.retire_execution(ids, states)
    }
}
