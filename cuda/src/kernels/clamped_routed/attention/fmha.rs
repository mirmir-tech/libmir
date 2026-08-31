use mircuda::{DeviceBuffer, FmhaBf16Plan, FmhaCausalWindow, LaunchConfig, Stream, bf16};

use super::{ClampedRoutedAttention, narrow};
use crate::{PagedPrefillBatch, Result, backend::WindowedPrefillStaging};

#[derive(Debug)]
pub(super) struct ClampedRoutedFmha {
    pub(super) plan: FmhaBf16Plan,
    pub(super) window: FmhaCausalWindow,
}

impl ClampedRoutedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_fmha(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        batch: &PagedPrefillBatch,
        tables: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        softmax_lse: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<bool> {
        let Some(fmha) = &self.fmha else {
            return Ok(false);
        };
        let max_context_tokens = batch.fmha_max_context_tokens();
        fmha.plan.execute_paged_varlen_windowed(
            stream,
            query,
            key_pages,
            value_pages,
            output,
            batch.query_starts(),
            batch.token_counts(),
            batch.context_starts(),
            tables,
            softmax_lse,
            batch.active(),
            batch.tokens(),
            batch.max_query_tokens(),
            max_context_tokens,
            batch.max_blocks(),
            batch.cache_config().block_size,
            fmha.window,
            scale,
        )?;
        self.sink_scale.launch(
            stream,
            LaunchConfig::for_elements(batch.tokens() * self.query_heads * self.head_dim, 256)?,
            (
                output,
                &*softmax_lse,
                sinks,
                narrow(batch.tokens())?,
                narrow(self.query_heads)?,
                narrow(self.head_dim)?,
            ),
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_windowed_fmha(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        batch: &PagedPrefillBatch,
        staged: &WindowedPrefillStaging,
        sinks: &DeviceBuffer<bf16>,
        softmax_lse: &mut DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<bool> {
        let Some(fmha) = &self.fmha else {
            return Ok(false);
        };
        fmha.plan.execute_paged_varlen_windowed(
            stream,
            query,
            staged.key_pages(),
            staged.value_pages(),
            output,
            batch.query_starts(),
            staged.token_counts(),
            staged.context_starts(),
            staged.tables(),
            softmax_lse,
            batch.active(),
            batch.tokens(),
            batch.max_query_tokens(),
            staged.fmha_max_context_tokens(),
            staged.blocks_per_row(),
            batch.cache_config().block_size,
            fmha.window,
            scale,
        )?;
        self.sink_scale.launch(
            stream,
            LaunchConfig::for_elements(batch.tokens() * self.query_heads * self.head_dim, 256)?,
            (
                output,
                &*softmax_lse,
                sinks,
                narrow(batch.tokens())?,
                narrow(self.query_heads)?,
                narrow(self.head_dim)?,
            ),
        )?;
        Ok(true)
    }
}
