//! Shape checks and positions of a packed prefill pass.

use runtime::kv::BlockTable;

use super::CudaSharedRoutedPrefillBatch;
use crate::{CudaSharedRoutedModelSession, Error, Result};

impl CudaSharedRoutedPrefillBatch {
    pub(super) fn paging_positions(&self, starts: &[usize]) -> Result<Vec<u32>> {
        starts
            .iter()
            .zip(&self.counts)
            .flat_map(|(start, count)| *start..*start + *count)
            .map(|value| {
                u32::try_from(value)
                    .map_err(|_| Error::InvalidPagedKv("shared-routed position exceeds u32"))
            })
            .collect()
    }

    pub(super) fn validate(
        &self,
        sessions: &[&mut CudaSharedRoutedModelSession],
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
    ) -> Result<()> {
        if sessions.len() != self.rows
            || tables.len() != self.rows
            || starts.len() != self.rows
            || tokens.len() != self.counts.iter().sum::<usize>()
        {
            return Err(Error::InvalidDecoderKernel("shared-routed prefill batch shape mismatch"));
        }
        for (session, start) in sessions.iter().zip(starts) {
            if session.position != *start {
                return Err(Error::InvalidPagedKv("shared-routed prefill position mismatch"));
            }
        }
        Ok(())
    }
}
