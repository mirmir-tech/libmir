use mircuda::{DeviceBuffer, KernelNode, LaunchConfig, Stream, bf16};

use super::{AttentionKernel, KvStoreKernel, PagedAttention, PagedKvStore};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, require},
};

type KvArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<u8>,
    &'a mut DeviceBuffer<u8>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
);

type AttentionArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    u32,
);

impl PagedKvStore {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_captured(
        &self,
        stream: &Stream,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
        key_pages: &mut DeviceBuffer<u8>,
        value_pages: &mut DeviceBuffer<u8>,
        local_start: usize,
        token_count: usize,
        physical_block: usize,
        page_start: usize,
    ) -> Result<KernelNode<KvStoreKernel>> {
        let (config, arguments) = self.launch(
            keys, values, key_pages, value_pages, local_start, token_count, physical_block,
            page_start,
        )?;
        Ok(self.kernel.launch_captured(stream, config, arguments)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch<'a>(
        &self,
        keys: &'a DeviceBuffer<bf16>,
        values: &'a DeviceBuffer<bf16>,
        key_pages: &'a mut DeviceBuffer<u8>,
        value_pages: &'a mut DeviceBuffer<u8>,
        local_start: usize,
        token_count: usize,
        physical_block: usize,
        page_start: usize,
    ) -> Result<(LaunchConfig, KvArguments<'a>)> {
        let width = self.spec.kv_heads * self.spec.key_head_dim.max(self.spec.value_head_dim);
        require(
            "paged KV keys",
            (local_start + token_count) * self.spec.kv_heads * self.spec.key_head_dim,
            keys.len(),
        )?;
        require(
            "paged KV values",
            (local_start + token_count) * self.spec.kv_heads * self.spec.value_head_dim,
            values.len(),
        )?;
        require("paged KV key pages", self.key_bytes()?, key_pages.len())?;
        require("paged KV value pages", self.value_bytes()?, value_pages.len())?;
        if physical_block >= self.spec.block_count
            || page_start + token_count > self.spec.block_size
        {
            return Err(Error::InvalidPagedKv("write exceeds a physical KV page"));
        }
        let threads = 256_usize;
        let config = LaunchConfig {
            grid: (narrow((token_count * width).div_ceil(threads))?, 1, 1),
            block: (narrow(threads)?, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok((
            config,
            (
                keys,
                values,
                key_pages,
                value_pages,
                narrow(local_start)?,
                narrow(token_count)?,
                narrow(physical_block)?,
                narrow(page_start)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.key_head_dim)?,
                narrow(self.spec.value_head_dim)?,
            ),
        ))
    }
}

impl PagedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_captured(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_table: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
        split_threshold: usize,
    ) -> Result<KernelNode<AttentionKernel>> {
        let (config, arguments) = self.launch(
            query, key_pages, value_pages, block_table, output, token_count, block_count, window,
            scale, split_threshold,
        )?;
        Ok(self.kernel.launch_captured(stream, config, arguments)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch<'a>(
        &self,
        query: &'a DeviceBuffer<bf16>,
        key_pages: &'a DeviceBuffer<u8>,
        value_pages: &'a DeviceBuffer<u8>,
        block_table: &'a DeviceBuffer<u32>,
        output: &'a mut DeviceBuffer<bf16>,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
        split_threshold: usize,
    ) -> Result<(LaunchConfig, AttentionArguments<'a>)> {
        require("paged attention query", self.spec.query_heads * self.spec.head_dim, query.len())?;
        require(
            "paged attention output",
            self.spec.query_heads * self.spec.value_head_dim,
            output.len(),
        )?;
        require("paged attention block table", self.spec.max_blocks, block_table.len())?;
        let capacity = block_count
            .checked_mul(self.spec.block_size)
            .ok_or(Error::InvalidPagedKv("paged attention capacity overflow"))?;
        if token_count == 0
            || block_count == 0
            || block_count > self.spec.max_blocks
            || token_count > capacity
            || !scale.is_finite()
        {
            return Err(Error::InvalidPagedKv("invalid paged attention execution geometry"));
        }
        let config = LaunchConfig {
            grid: (narrow(self.spec.query_heads)?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        Ok((
            config,
            (
                query,
                key_pages,
                value_pages,
                block_table,
                output,
                narrow(token_count)?,
                narrow(block_count)?,
                narrow(self.spec.block_size)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(window.unwrap_or(0))?,
                scale,
                narrow(split_threshold)?,
            ),
        ))
    }
}
