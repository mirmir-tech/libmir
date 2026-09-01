use mircuda::{DeviceBuffer, bf16};

use super::ClampedRoutedExecutionPlan;
use crate::{
    CudaClampedRoutedModelTemplate, Error, PagedPrefillBatch, Result,
    backend::clamped_routed::{
        projection::ClampedRoutedEmbedding, session::ClampedRoutedSessionState,
    },
    kernels::ClampedRoutedBatchSplitDecode,
};

impl ClampedRoutedExecutionPlan {
    pub(in crate::backend::clamped_routed) fn execute_batch(
        &mut self,
        template: &CudaClampedRoutedModelTemplate,
        state: &mut ClampedRoutedSessionState,
        embedding: &ClampedRoutedEmbedding,
        batch: &PagedPrefillBatch,
    ) -> Result<&DeviceBuffer<bf16>> {
        embedding.execute_batch(
            &self.token_ids,
            self.tokens,
            &template.embedding,
            &mut self.first,
        )?;
        if self.layers.len() != state.caches.len() || batch.tokens() != self.tokens {
            return Err(Error::InvalidPagedKv(
                "packed clamped-routed cache or token geometry differs",
            ));
        }
        self.prepare_windowed_prefill(template, batch)?;
        if batch.max_query_tokens() == 1 && self.batch_split_decode.is_none() {
            self.batch_split_decode = ClampedRoutedBatchSplitDecode::compile(
                &template.backend,
                template.config.storage(template.cache),
                template.config.query_heads,
                template.max_sequence_blocks,
                batch.active(),
            )?;
        }
        let first = self
            .layers
            .first()
            .ok_or(Error::InvalidExecutionPlan("decoder has no layers"))?;
        first.prepare_rope(batch.positions(), &mut self.scratch)?;
        let (layers, scratch, dense_experts, batch_split_decode, windowed_prefill) = (
            &mut self.layers,
            &mut self.scratch,
            &mut self.dense_experts,
            &mut self.batch_split_decode,
            &mut self.windowed_prefill,
        );
        for (index, (layer, cache)) in layers.iter_mut().zip(&mut state.caches).enumerate() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            layer.execute_batch(
                &template.layers[index],
                input,
                cache,
                batch,
                windowed_prefill.as_mut(),
                scratch,
                dense_experts,
                batch_split_decode,
                output,
            )?;
            #[cfg(feature = "diagnostics")]
            {
                self.fingerprints.record(&scratch.residual, index * 2)?;
                self.fingerprints.record(output, index * 2 + 1)?;
            }
        }
        Ok(self.hidden())
    }
}
