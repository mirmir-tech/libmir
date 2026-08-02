use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;

use super::PagedAttentionBf16;
use crate::{Error, PagedKvCache, Result};

#[derive(Debug)]
pub(super) struct FmhaPageWorkspace {
    keys: DeviceBuffer<u8>,
    values: DeviceBuffer<u8>,
    capacity_tokens: usize,
}

pub(super) fn contiguous_page_token(table: &BlockTable, tokens: usize) -> Option<usize> {
    let block_size = table.block_size()?;
    let needed = tokens.div_ceil(block_size);
    let blocks = table.blocks().get(..needed)?;
    let first = usize::try_from(blocks.first()?.0).ok()?;
    let contiguous = blocks
        .iter()
        .enumerate()
        .all(|(index, block)| usize::try_from(block.0).ok() == first.checked_add(index));
    contiguous.then(|| first.checked_mul(block_size)).flatten()
}

impl PagedAttentionBf16 {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_gathered_fmha(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        output: &mut DeviceBuffer<bf16>,
        query_tokens: usize,
        context_tokens: usize,
        block_count: usize,
        scale: f32,
    ) -> Result<()> {
        self.ensure_fmha_workspace(context_tokens)?;
        let gather = self
            .gather
            .as_ref()
            .ok_or(Error::InvalidExecutionPlan("missing paged BF16 gather plan"))?;
        let workspace = self
            .fmha_workspace
            .as_mut()
            .ok_or(Error::InvalidExecutionPlan("missing paged BF16 FMHA workspace"))?;
        gather.execute(
            &self.stream,
            cache.key_pages(),
            cache.value_pages(),
            &self.table_device,
            &mut workspace.keys,
            &mut workspace.values,
            context_tokens,
            block_count,
        )?;
        Ok(self
            .fmha
            .as_ref()
            .ok_or(Error::InvalidExecutionPlan("missing BF16 FMHA plan"))?
            .execute(
                &self.stream,
                query,
                &workspace.keys,
                &workspace.values,
                output,
                query_tokens,
                context_tokens,
                0,
                scale,
            )?)
    }

    fn ensure_fmha_workspace(&mut self, context_tokens: usize) -> Result<()> {
        if self
            .fmha_workspace
            .as_ref()
            .is_some_and(|workspace| workspace.capacity_tokens >= context_tokens)
        {
            return Ok(());
        }
        let key_bytes =
            workspace_bytes(context_tokens, self.storage.kv_heads, self.storage.key_head_dim)?;
        let value_bytes =
            workspace_bytes(context_tokens, self.storage.kv_heads, self.storage.value_head_dim)?;
        self.fmha_workspace = Some(FmhaPageWorkspace {
            keys: self.pool.allocate(&self.stream, key_bytes)?,
            values: self.pool.allocate(&self.stream, value_bytes)?,
            capacity_tokens: context_tokens,
        });
        Ok(())
    }
}

fn workspace_bytes(tokens: usize, heads: usize, dimensions: usize) -> Result<usize> {
    tokens
        .checked_mul(heads)
        .and_then(|elements| elements.checked_mul(dimensions))
        .and_then(|elements| elements.checked_mul(size_of::<bf16>()))
        .ok_or(Error::InvalidExecutionPlan("paged BF16 FMHA workspace overflow"))
}
