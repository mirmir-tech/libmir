use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::{
    CudaSharedRoutedLayerState, CudaSharedRoutedModelTemplate, SharedRoutedLayerTemplate,
    boundary::SharedRoutedEmbedding,
};
use crate::{
    CudaAffineGatedDeltaMoeExecution, CudaAffineGatedFullAttentionMoeExecution, Error, Result,
    kernels::VisionEmbeddingSplice,
};

#[derive(Debug)]
enum SharedRoutedLayerExecution {
    Linear(Box<CudaAffineGatedDeltaMoeExecution>),
    Full(Box<CudaAffineGatedFullAttentionMoeExecution>),
}

#[derive(Debug)]
pub(super) struct SharedRoutedExecutionPlan {
    tokens: usize,
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
        let elements = tokens
            .checked_mul(template.decoder.hidden_size)
            .ok_or(Error::InvalidDecoderKernel("shared-routed activation size overflow"))?;
        let position_elements = tokens
            .checked_mul(3)
            .ok_or(Error::InvalidDecoderKernel("shared-routed position size overflow"))?;
        let layers = template
            .layers
            .iter()
            .map(|layer| match layer {
                SharedRoutedLayerTemplate::Linear(layer) => {
                    layer.prepare(tokens).map(Box::new).map(SharedRoutedLayerExecution::Linear)
                },
                SharedRoutedLayerTemplate::Full(layer) => {
                    layer.prepare(tokens).map(Box::new).map(SharedRoutedLayerExecution::Full)
                },
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tokens,
            token_staging: backend.inner.context.allocate_pinned(tokens)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            position_staging: backend.inner.context.allocate_pinned(position_elements)?,
            positions: backend.inner.pool.allocate(&backend.inner.stream, position_elements)?,
            first: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            second: backend.inner.pool.allocate(&backend.inner.stream, elements)?,
            layers,
        })
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
        for (index, (layer, state)) in self.layers.iter_mut().zip(states).enumerate() {
            let (input, output) = if index.is_multiple_of(2) {
                (&self.first, &mut self.second)
            } else {
                (&self.second, &mut self.first)
            };
            match (layer, state) {
                (
                    SharedRoutedLayerExecution::Linear(layer),
                    CudaSharedRoutedLayerState::Linear(state),
                ) => {
                    if state.offset() != start_position {
                        return Err(Error::InvalidDecoderKernel(
                            "Gated Delta session position mismatch",
                        ));
                    }
                    layer.execute(input, state, output)?;
                },
                (
                    SharedRoutedLayerExecution::Full(layer),
                    CudaSharedRoutedLayerState::Full(state),
                ) => {
                    let write = KvWritePlan::prefill(
                        session_id, index, table, start_position, self.tokens,
                    )?;
                    layer.execute_with_image_span(
                        input, &self.positions, state, &write, table, start_position, None,
                        image_span, output,
                    )?;
                },
                _ => {
                    return Err(Error::InvalidDecoderKernel(
                        "shared-routed execution and state kinds differ",
                    ));
                },
            }
        }
        Ok(if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        })
    }
}
