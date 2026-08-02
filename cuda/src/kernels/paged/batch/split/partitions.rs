use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{BatchedSplitAttentionWorkspace, BatchedSplitPagedAttention};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product},
};

impl BatchedSplitPagedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_partitions(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>,
        token_counts: &DeviceBuffer<u32>,
        block_counts: &DeviceBuffer<u32>,
        workspace: &mut BatchedSplitAttentionWorkspace,
        output: &DeviceBuffer<bf16>,
        batch_size: usize,
        window: Option<usize>,
        scale: f32,
        minimum_tokens: usize,
        maximum_tokens: usize,
    ) -> Result<()> {
        self.validate(
            query, block_tables, token_counts, block_counts, workspace, output, batch_size,
        )?;
        let visible = window.map_or(maximum_tokens, |tokens| maximum_tokens.min(tokens));
        let launch_partitions = visible.div_ceil(self.partition_tokens).min(self.max_partitions);
        if launch_partitions == 0 {
            return Err(Error::InvalidPagedKv("batched split attention has no visible tokens"));
        }
        Ok(self.split.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(product(self.spec.kv_heads, launch_partitions)?)?,
                    narrow(batch_size)?,
                    1,
                ),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key_pages,
                value_pages,
                block_tables,
                token_counts,
                block_counts,
                &mut workspace.values,
                &mut workspace.maxima,
                &mut workspace.denominators,
                narrow(batch_size)?,
                narrow(self.spec.max_blocks)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
                narrow(self.partition_tokens)?,
                narrow(launch_partitions)?,
                narrow(self.max_partitions)?,
                narrow(minimum_tokens)?,
            ),
        )?)
    }
}
