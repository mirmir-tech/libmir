mod cache;
mod layer;

use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::BlockTable;
use uuid::Uuid;

pub(super) use self::cache::SharedRoutedExecutionPlanCache;
use self::layer::{PlanStep, SharedRoutedLayerExecution};
use super::{
    CudaSharedRoutedLayerState, CudaSharedRoutedModelTemplate, boundary::SharedRoutedEmbedding,
};
use crate::{Error, Result, kernels::VisionEmbeddingSplice};

#[derive(Debug)]
pub(super) struct SharedRoutedExecutionPlan {
    tokens: usize,
    /// Pool bytes this plan took at construction.
    bytes: u64,
    token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    position_staging: PinnedBuffer<u32>,
    positions: DeviceBuffer<u32>,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    layers: Vec<SharedRoutedLayerExecution>,
}

impl SharedRoutedExecutionPlan {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, tokens: usize) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("empty shared-routed execution plan"));
        }
        let backend = &template.backend;
        let used_before = backend.pool().stats()?.used;
        let elements = tokens
            .checked_mul(template.decoder.hidden_size)
            .ok_or(Error::InvalidDecoderKernel("shared-routed activation size overflow"))?;
        let position_elements = tokens
            .checked_mul(3)
            .ok_or(Error::InvalidDecoderKernel("shared-routed position size overflow"))?;
        let layers = template
            .layers
            .iter()
            .map(|layer| SharedRoutedLayerExecution::new(layer, tokens))
            .collect::<Result<Vec<_>>>()?;
        let mut plan = Self {
            tokens,
            bytes: 0,
            token_staging: backend.inner.context.allocate_pinned(tokens)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            position_staging: backend.inner.context.allocate_pinned(position_elements)?,
            positions: backend.inner.pool.allocate(&backend.inner.stream, position_elements)?,
            first: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            second: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            layers,
        };
        plan.bytes = backend.pool().stats()?.used.saturating_sub(used_before);
        Ok(plan)
    }

    pub(super) fn upload(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<()> {
        if tokens.len() != self.tokens || positions.len() != 3 * self.tokens {
            return Err(Error::InvalidDecoderKernel("shared-routed plan upload shape mismatch"));
        }
        self.token_staging.copy_from_slice(tokens)?;
        self.position_staging.copy_from_slice(positions)?;
        let stream = &template.backend.inner.stream;
        stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        stream.copy_to_device(&mut self.position_staging, &mut self.positions)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
        embedding: &SharedRoutedEmbedding,
        states: &mut [CudaSharedRoutedLayerState],
        session_id: Uuid,
        table: &BlockTable,
        start_position: usize,
        image_span: Option<(usize, usize)>,
        image: Option<&DeviceBuffer<bf16>>,
    ) -> Result<&DeviceBuffer<bf16>> {
        embedding.execute_batch(&self.token_ids, self.tokens, &mut self.first)?;
        if let Some(image) = image {
            let (start, end) = image_span.ok_or(Error::InvalidVisionKernel(
                "shared-routed image embedding has no prompt span",
            ))?;
            if start >= end || end > self.tokens {
                return Err(Error::InvalidVisionKernel("invalid shared-routed image prompt span"));
            }
            VisionEmbeddingSplice::compile(
                &template.backend.inner.compiler,
                end - start,
                template.decoder.hidden_size,
                start,
            )?
            .execute(&template.backend.inner.stream, image, &mut self.first)?;
        }
        if self.layers.len() != states.len() {
            return Err(Error::InvalidDecoderKernel("shared-routed layer state count mismatch"));
        }
        let step = PlanStep {
            session_id,
            table,
            start_position,
            tokens: self.tokens,
            positions: &self.positions,
            image_span,
        };
        for (index, (layer, state)) in self.layers.iter_mut().zip(states).enumerate() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            layer.execute(index, state, &step, input, output)?;
        }
        Ok(if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        })
    }
}
