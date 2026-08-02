use mircuda::{DeviceBuffer, Stream, bf16};

use super::{SplitAttentionWorkspace, SplitPagedAttention};
use crate::Result;

impl SplitPagedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_partitions(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_table: &DeviceBuffer<u32>,
        workspace: &mut SplitAttentionWorkspace,
        output: &DeviceBuffer<bf16>,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<usize> {
        let active = self.validate(
            query, block_table, workspace, output, token_count, block_count, window, scale,
        )?;
        self.split.launch(
            stream,
            self.configs(active)?.split,
            self.split_arguments(
                query, key_pages, value_pages, block_table, workspace, token_count, block_count,
                window, scale, active, 0,
            )?,
        )?;
        Ok(active)
    }

    #[must_use]
    pub(crate) const fn max_partitions(&self) -> usize {
        self.max_partitions
    }
}
