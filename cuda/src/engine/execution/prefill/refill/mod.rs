use std::collections::HashSet;

use runtime::{
    backend::{ModelHandle, PrefillRequest},
    progress::ProgressEvent,
};

use super::{CudaEngine, CudaPrefillBatch};
use crate::{Error, Result};

impl CudaEngine {
    /// Adds rows at a completed step boundary. Existing session state stays
    /// resident and the caller must preserve capacity and token-budget bounds.
    pub fn extend_prefill_batch(
        &self,
        batch: &mut CudaPrefillBatch,
        requests: &[PrefillRequest],
        progress: &mut dyn FnMut(usize, ProgressEvent),
    ) -> Result<()> {
        let first = batch
            .sequences
            .first()
            .ok_or(Error::InvalidDecoderKernel("empty refill target"))?;
        validate(
            &first.request.model,
            batch.sequences.iter().map(|row| row.request.session_id),
            requests,
        )?;
        let incoming = self.prepare_prefill_batch(requests, progress)?;
        prepend(batch, incoming);
        Ok(())
    }
}

fn prepend(batch: &mut CudaPrefillBatch, incoming: CudaPrefillBatch) {
    batch.scheduled_tokens = batch.scheduled_tokens.saturating_add(incoming.scheduled_tokens);
    batch.sequences.splice(..0, incoming.sequences);
    batch.cursor = 0;
}

fn validate(
    model: &ModelHandle,
    sessions: impl Iterator<Item = uuid::Uuid>,
    requests: &[PrefillRequest],
) -> Result<()> {
    let mut unique = sessions.collect::<HashSet<_>>();
    if requests.is_empty()
        || requests.iter().any(|row| {
            row.model.id != model.id
                || row.model.backend != model.backend
                || !unique.insert(row.session_id)
        })
    {
        return Err(Error::InvalidDecoderKernel("invalid CUDA prefill refill sessions"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
