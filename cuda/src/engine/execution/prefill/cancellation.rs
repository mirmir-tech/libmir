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
        // Synchronizing a stream while another model captures a graph is invalid.
        // Release admission before release_session acquires its model runner.
        let execution = self.execution.acquire_prefill()?;
        self.backend.synchronize()?;
        drop(execution);
        for session in sessions {
            loaded.release_session(*session)?;
        }
        batch.sequences.retain(|row| !sessions.contains(&row.request.session_id));
        batch.cursor %= batch.sequences.len().max(1);
        Ok(())
    }
}
