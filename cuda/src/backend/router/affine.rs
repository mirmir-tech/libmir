use mircuda::{DeviceBuffer, Stream, bf16};

use super::RouterSelection;
use crate::{
    AffineQuantizedBf16Qmm, AffineQuantizedConfig, AffineQuantizedTensors, CudaBackend, Error,
    Result,
    kernels::{RouterUnitSpec, RouterUnitTopK},
};

#[derive(Debug)]
pub struct AffineRouterBf16 {
    projection: AffineQuantizedBf16Qmm,
    top_k: RouterUnitTopK,
    stream: Stream,
    scores: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    weights: DeviceBuffer<bf16>,
}

impl CudaBackend {
    pub fn prepare_affine_router_bf16(
        &self,
        tokens: usize,
        projection: AffineQuantizedConfig,
        top_k: usize,
    ) -> Result<AffineRouterBf16> {
        AffineRouterBf16::new(self, tokens, projection, top_k)
    }
}

impl AffineRouterBf16 {
    fn new(
        backend: &CudaBackend,
        tokens: usize,
        projection: AffineQuantizedConfig,
        top_k: usize,
    ) -> Result<Self> {
        let selections = tokens
            .checked_mul(top_k)
            .ok_or(Error::InvalidRouter("affine router selection overflow"))?;
        let scores = tokens
            .checked_mul(projection.output_features)
            .ok_or(Error::InvalidRouter("affine router score overflow"))?;
        Ok(Self {
            projection: backend.prepare_affine_quantized_bf16_qmm(tokens, projection, 1)?,
            top_k: RouterUnitTopK::compile(
                &backend.inner.compiler,
                RouterUnitSpec {
                    tokens,
                    experts: projection.output_features,
                    top_k,
                },
            )?,
            stream: backend.inner.stream.clone(),
            scores: backend.inner.pool.allocate(&backend.inner.stream, scores)?,
            selected: backend.inner.pool.allocate(&backend.inner.stream, selections)?,
            weights: backend.inner.pool.allocate(&backend.inner.stream, selections)?,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        projection: AffineQuantizedTensors<'_>,
    ) -> Result<RouterSelection<'_>> {
        self.projection.execute(input, projection, &mut self.scores, 0)?;
        self.top_k
            .execute(&self.stream, &self.scores, &mut self.selected, &mut self.weights)?;
        Ok(RouterSelection {
            indices: &self.selected,
            weights: &self.weights,
        })
    }
}
