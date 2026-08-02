use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;

use super::{CapturedPagedAttentionNodes, PagedAttentionBf16};
use crate::{
    PagedKvCache, Result,
    kernels::{MergeAttentionArguments, SplitAttentionArguments},
};

impl PagedAttentionBf16 {
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
