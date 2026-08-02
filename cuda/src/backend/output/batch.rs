use mircuda::{DeviceBuffer, Stream, bf16};

use super::weight::CudaOutputHeadTemplate;
use crate::{
    Bf16Projection, CudaBackend, CudaTensor, DensePlanRequest, DenseRole, ExecutionPhase, Result,
};

/// Exact BF16 output projection prepared for one fixed decode bucket.
#[derive(Debug)]
pub struct CudaBatchOutputHead {
    projection: Bf16Projection,
    weight: CudaTensor,
    stream: Stream,
}

impl CudaBatchOutputHead {
    pub(crate) fn new(
        backend: &CudaBackend,
        template: &CudaOutputHeadTemplate,
        rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            projection: backend.prepare_bf16_projection(DensePlanRequest {
                phase: ExecutionPhase::Decode,
                role: DenseRole::OutputHead,
                tokens: rows,
                input_features: template.input_features,
                output_features: template.output_features,
            })?,
            weight: template.exact.clone(),
            stream: backend.inner.stream.clone(),
        })
    }

    pub(crate) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.projection.execute(input, &self.weight, output)
    }

    pub(crate) const fn stream(&self) -> &Stream {
        &self.stream
    }
}
