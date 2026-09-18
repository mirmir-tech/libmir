use runtime::backend::PrefillOutput;

use super::{CudaEngine, CudaPrefillBatch};
use crate::Result;

impl CudaEngine {
    /// Publishes ready rows while preserving pending rows and their device
    /// state.
    pub fn take_completed_prefill_rows(
        &self,
        batch: &mut CudaPrefillBatch,
    ) -> Result<Vec<(uuid::Uuid, PrefillOutput)>> {
        if self.prefill_schedule(&batch.model_id)? != crate::CudaPrefillSchedule::CompletionFirst
            || !batch.sequences.iter().any(|row| !row.pending())
        {
            return Ok(Vec::new());
        }
        let loaded = self.model(&batch.model_id)?;
        let ready = batch.sequences.extract_if(.., |row| !row.pending()).collect::<Vec<_>>();
        batch.cursor = 0;
        ready
            .into_iter()
            .map(|row| {
                let session = row.request.session_id;
                Ok((session, row.finish(&loaded, batch.started)?))
            })
            .collect()
    }
}
