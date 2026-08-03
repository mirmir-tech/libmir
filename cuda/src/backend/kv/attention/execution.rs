use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;

use super::{PagedAttentionBf16, pages::contiguous_page_token};
use crate::{PagedKvCache, Result};

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
        self.execute_prefill_masked(
            query, cache, table, output, query_tokens, start_position, window, scale, None,
        )
    }

    /// Executes causal prefill while opening one bidirectional image block.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_prefill_masked(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        start_position: usize,
        window: Option<usize>,
        scale: f32,
        image: Option<(usize, usize)>,
    ) -> Result<()> {
        self.validate(cache, table)?;
        self.update_table(table)?;
        let context = start_position
            .checked_add(query_tokens)
            .ok_or(crate::Error::InvalidPagedKv("prefill attention context overflow"))?;
        if window.is_none() && image.is_none() {
            if let Some(fmha) = &mut self.paged_fmha {
                return fmha.execute(
                    query,
                    cache.key_pages(),
                    cache.value_pages(),
                    output,
                    &self.table_device,
                    query_tokens,
                    context,
                    scale,
                );
            }
            if let (Some(first_page_token), Some(fmha)) =
                (contiguous_page_token(table, context), &self.fmha)
            {
                return Ok(fmha.execute(
                    &self.stream,
                    query,
                    cache.key_pages(),
                    cache.value_pages(),
                    output,
                    query_tokens,
                    context,
                    first_page_token,
                    scale,
                )?);
            }
            if self.fmha.is_some() {
                return self.execute_gathered_fmha(
                    query,
                    cache,
                    output,
                    query_tokens,
                    context,
                    table.blocks().len(),
                    scale,
                );
            }
        }
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
            image,
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
        self.ensure_tuned(query, cache, table, output, window, scale);
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
}
