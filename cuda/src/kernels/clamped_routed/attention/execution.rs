use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::ClampedRoutedAttention;
use crate::{PagedPrefillBatch, Result};

impl ClampedRoutedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_prefill_batch(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        current_keys: &DeviceBuffer<bf16>,
        current_values: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        batch: &PagedPrefillBatch,
        tables: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        softmax_lse: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        if window.is_none()
            && self.execute_fmha(
                stream, query, key_pages, value_pages, batch, tables, sinks, softmax_lse, output,
                scale,
            )?
        {
            return Ok(());
        }
        Ok(self.batch_prefill.launch(
            stream,
            Self::launch(batch.tokens() * self.query_heads, self.head_dim)?,
            (
                query,
                key_pages,
                current_keys,
                current_values,
                value_pages,
                tables,
                batch.request_indices(),
                batch.positions(),
                batch.query_starts(),
                batch.block_counts(),
                sinks,
                output,
                narrow(batch.tokens())?,
                narrow(batch.max_blocks())?,
                narrow(self.block_size)?,
                narrow(self.query_heads)?,
                narrow(self.kv_heads)?,
                narrow(self.head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        current_keys: &DeviceBuffer<bf16>,
        current_values: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        table: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
        blocks: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        Ok(self.decode.launch(
            stream,
            Self::launch(self.query_heads, self.head_dim)?,
            (
                query,
                current_keys,
                current_values,
                key_pages,
                value_pages,
                table,
                sinks,
                output,
                narrow(tokens)?,
                narrow(blocks)?,
                narrow(self.block_size)?,
                narrow(self.query_heads)?,
                narrow(self.kv_heads)?,
                narrow(self.head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_prefill(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        current_keys: &DeviceBuffer<bf16>,
        current_values: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        table: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        start: usize,
        blocks: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        Ok(self.prefill.launch(
            stream,
            Self::launch(query_tokens * self.query_heads, self.head_dim)?,
            (
                query,
                current_keys,
                current_values,
                key_pages,
                value_pages,
                table,
                sinks,
                output,
                narrow(query_tokens)?,
                narrow(start)?,
                narrow(blocks)?,
                narrow(self.block_size)?,
                narrow(self.query_heads)?,
                narrow(self.kv_heads)?,
                narrow(self.head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
            ),
        )?)
    }

    pub(super) fn launch(blocks: usize, head_dim: usize) -> Result<LaunchConfig> {
        let threads = if head_dim == 64 {
            32
        } else {
            head_dim.next_multiple_of(32).min(256)
        };
        Ok(LaunchConfig {
            grid: (narrow(blocks)?, 1, 1),
            block: (narrow(threads)?, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}

pub(super) fn narrow(value: usize) -> Result<u32> {
    Ok(u32::try_from(value)?)
}
