use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, CudaAffineGatedDeltaExecution,
};
use crate::{CudaBackend, CudaTensor, Error, Result, kernels::GatedDeltaAlphaBeta};

pub(super) fn prepare_dense_alpha_beta(
    backend: &CudaBackend,
    config: AffineGatedDeltaLayerConfig,
    weights: &AffineGatedDeltaLayerWeights,
    tokens: usize,
) -> Result<Option<GatedDeltaAlphaBeta>> {
    weights
        .alpha
        .dense_bf16()
        .zip(weights.beta.dense_bf16())
        .map(|_| {
            GatedDeltaAlphaBeta::compile(
                &backend.inner.compiler,
                tokens,
                config.hidden_size,
                config.value_heads,
            )
        })
        .transpose()
}

impl CudaAffineGatedDeltaExecution {
    pub(super) fn project_alpha_beta(&mut self, input: &DeviceBuffer<bf16>) -> Result<()> {
        if let Some(operation) = &self.dense_alpha_beta {
            return operation.execute(
                &self.backend.inner.stream,
                input,
                bf16(self.weights.alpha.dense_bf16())?,
                bf16(self.weights.beta.dense_bf16())?,
                &mut self.scratch.alpha,
                &mut self.scratch.beta,
            );
        }
        self.alpha
            .as_mut()
            .ok_or(Error::InvalidExecutionPlan("Gated Delta alpha projection is missing"))?
            .execute(input, &mut self.scratch.alpha)?;
        self.beta
            .as_mut()
            .ok_or(Error::InvalidExecutionPlan("Gated Delta beta projection is missing"))?
            .execute(input, &mut self.scratch.beta)
    }
}

fn bf16(tensor: Option<&CudaTensor>) -> Result<&DeviceBuffer<bf16>> {
    tensor
        .and_then(CudaTensor::as_bf16)
        .ok_or(Error::InvalidExecutionPlan("dense Gated Delta gate weight is missing"))
}
