//! Decode attention through `FlashAttention` when the layer has a plan.

use super::{ClampedRoutedLayerExecution, execution::bf16s};
use crate::{
    PagedKvCache, PagedPrefillBatch, Result,
    backend::clamped_routed::{scratch::ClampedRoutedScratch, weights::ClampedRoutedLayerWeights},
    kernels::ClampedRoutedBatchSplitDecode,
};

impl ClampedRoutedLayerExecution {
    /// Decode attention through `FlashAttention` with sinks, `false` when
    /// the batch is not a decode batch or no plan applies. A windowed layer
    /// stores the new K/V in its ring first: the slot it overwrites lies
    /// before the window, and `FlashAttention` then reads every key it keeps
    /// from the ring.
    pub(super) fn decode_fmha(
        &self,
        weights: &ClampedRoutedLayerWeights,
        cache: &mut PagedKvCache,
        batch: &PagedPrefillBatch,
        scratch: &mut ClampedRoutedScratch,
        batch_split_decode: &mut Option<ClampedRoutedBatchSplitDecode>,
    ) -> Result<bool> {
        let Some(split) = batch_split_decode.as_mut() else {
            return Ok(false);
        };
        if batch.max_query_tokens() != 1
            || !self.attention.has_fmha()
            || !split.fmha.fits(batch.active())
        {
            return Ok(false);
        }
        let tables = if cache.is_windowed() {
            cache.store_prefill_batch(batch, &scratch.key, &scratch.value)?;
            batch.ring_tables()
        } else {
            batch.tables()
        };
        let decoded = self.attention.execute_fmha_decode(
            &self.stream,
            &scratch.query,
            cache.key_pages(),
            cache.value_pages(),
            batch,
            tables,
            bf16s(&weights.sinks)?,
            &mut split.fmha,
            &mut scratch.attended,
            self.config.scale,
        )?;
        if !decoded && cache.is_windowed() {
            return Err(crate::Error::InvalidExecutionPlan(
                "windowed decode stored its K/V for a FlashAttention plan it could not run",
            ));
        }
        Ok(decoded)
    }
}
