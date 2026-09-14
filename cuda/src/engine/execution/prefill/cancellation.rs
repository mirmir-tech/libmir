use super::{CudaEngine, CudaPrefillBatch};
use crate::Result;

impl CudaEngine {
    /// Removes cancelled rows after settling their submitted CUDA work.
    pub fn cancel_prefill_sessions(
        &self,
        batch: &mut CudaPrefillBatch,
        sessions: &[uuid::Uuid],
    ) -> Result<()> {
        let loaded = self.model(&batch.model_id)?;
        self.backend.synchronize()?;
        for session in sessions {
            loaded.release_session(*session)?;
        }
        batch.sequences.retain(|row| !sessions.contains(&row.request.session_id));
        batch.cursor %= batch.sequences.len().max(1);
        Ok(())
    }
}
