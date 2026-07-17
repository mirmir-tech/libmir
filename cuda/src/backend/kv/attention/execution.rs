use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;

use super::{CapturedPagedAttentionNodes, PagedAttentionBf16};
use crate::{
    PagedKvCache, Result,
    kernels::{MergeAttentionArguments, SplitAttentionArguments},
};

impl PagedAttentionBf16 {
    /// Executes causal attention for every query token in a contiguous chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_prefill(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        start_position: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        self.validate(cache, table)?;
        self.update_table(table)?;
        self.prefill.execute(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            output,
            query_tokens,
            start_position,
            table.blocks().len(),
            window,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        let token_count = table.token_len();
        let visible = window.map_or(token_count, |limit| token_count.min(limit));
        if visible >= usize::try_from(self.split_threshold)? {
            return self.execute_split(query, cache, table, output, window, scale);
        }
        self.execute_direct(query, cache, table, output, window, scale)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_direct(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        self.validate(cache, table)?;
        self.update_table(table)?;
        self.operation.execute(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            output,
            table.token_len(),
            table.blocks().len(),
            window,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_split(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<()> {
        self.validate(cache, table)?;
        self.update_table(table)?;
        self.split.execute(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            &mut self.split_workspace,
            output,
            table.token_len(),
            table.blocks().len(),
            window,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_captured(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<CapturedPagedAttentionNodes> {
        self.validate(cache, table)?;
        self.update_table(table)?;
        let direct = self.operation.execute_captured(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            output,
            table.token_len(),
            table.blocks().len(),
            window,
            scale,
            usize::try_from(self.split_threshold)?,
        )?;
        let split = self.split.execute_captured(
            &self.stream,
            query,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            &mut self.split_workspace,
            output,
            table.token_len(),
            table.blocks().len(),
            window,
            scale,
            usize::try_from(self.split_threshold)?,
        )?;
        Ok(CapturedPagedAttentionNodes { direct, split })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn captured_split_arguments<'a>(
        &'a mut self,
        query: &'a DeviceBuffer<bf16>,
        cache: &'a PagedKvCache,
        token_count: u32,
        block_count: u32,
        window: u32,
        scale: f32,
    ) -> Result<SplitAttentionArguments<'a>> {
        let active = self.split.active_partitions(token_count, window)?;
        self.split.split_arguments(
            query,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            &mut self.split_workspace,
            usize::try_from(token_count)?,
            usize::try_from(block_count)?,
            window_option(window)?,
            scale,
            usize::try_from(active)?,
            usize::try_from(self.split_threshold)?,
        )
    }

    pub(in crate::backend) fn captured_merge_arguments<'a>(
        &'a self,
        output: &'a mut DeviceBuffer<bf16>,
        token_count: u32,
        window: u32,
    ) -> Result<MergeAttentionArguments<'a>> {
        let active = self.split.active_partitions(token_count, window)?;
        let visible_tokens = token_count.min(if window == 0 {
            token_count
        } else {
            window
        });
        self.split.merge_arguments(
            &self.split_workspace,
            output,
            usize::try_from(visible_tokens)?,
            usize::try_from(active)?,
            usize::try_from(self.split_threshold)?,
        )
    }
}

fn window_option(window: u32) -> Result<Option<usize>> {
    Ok((window > 0).then(|| usize::try_from(window)).transpose()?)
}
