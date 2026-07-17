use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::PagedKvStore;
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

impl PagedKvStore {
    #[allow(clippy::too_many_arguments)]
    pub fn execute_batch(
        &self,
        stream: &Stream,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
        key_pages: &mut DeviceBuffer<u8>,
        value_pages: &mut DeviceBuffer<u8>,
        block_tables: &DeviceBuffer<u32>,
        token_counts: &DeviceBuffer<u32>,
        batch_size: usize,
        max_blocks: usize,
    ) -> Result<()> {
        let key_width = product(self.spec.kv_heads, self.spec.key_head_dim)?;
        let value_width = product(self.spec.kv_heads, self.spec.value_head_dim)?;
        require("batched KV keys", product(batch_size, key_width)?, keys.len())?;
        require("batched KV values", product(batch_size, value_width)?, values.len())?;
        require("batched KV key pages", self.key_bytes()?, key_pages.len())?;
        require("batched KV value pages", self.value_bytes()?, value_pages.len())?;
        require("batched KV block tables", product(batch_size, max_blocks)?, block_tables.len())?;
        require("batched KV token counts", batch_size, token_counts.len())?;
        if batch_size == 0 || max_blocks == 0 {
            return Err(Error::InvalidPagedKv("invalid batched KV store geometry"));
        }
        let elements = product(batch_size, key_width.max(value_width))?;
        let threads = 256_usize;
        Ok(self.batch_kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                keys,
                values,
                key_pages,
                value_pages,
                block_tables,
                token_counts,
                narrow(batch_size)?,
                narrow(max_blocks)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.key_head_dim)?,
                narrow(self.spec.value_head_dim)?,
            ),
        )?)
    }
}
