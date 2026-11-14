//! Memory of the shared-routed template: pool usage, plan retention, and
//! the variants over another cache size or decode width.

use super::CudaSharedRoutedModelTemplate;
use crate::{Error, Result};

impl CudaSharedRoutedModelTemplate {
    pub(crate) fn pool_used_bytes(&self) -> Result<u64> {
        Ok(self.backend.pool().stats()?.used)
    }

    /// Pool bytes the retained scalar plans may still grow by, and the bytes
    /// one plan token cost so far.
    pub(crate) fn plan_retention(&self) -> Result<(u64, u64)> {
        let plans = self
            .plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        Ok((plans.headroom_bytes(), plans.bytes_per_token()))
    }

    /// Rows the decode batches' shared packed states are allocated for:
    /// the widest batch the scheduler forms.
    #[must_use]
    pub fn with_decode_rows(mut self, rows: usize) -> Self {
        self.decode_rows = rows.max(1);
        self
    }

    /// The same template over a K/V cache of `block_count` blocks. Sequence
    /// capacity follows the context length and block size, which are kept.
    #[must_use]
    pub fn with_cache_blocks(&self, block_count: u32) -> Self {
        let mut template = self.clone();
        template.cache.block_count = block_count;
        template
    }
}
