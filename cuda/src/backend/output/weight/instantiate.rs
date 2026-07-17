use super::CudaOutputHeadTemplate;
use crate::{
    CudaBackend, CudaOutputHead, DensePlanRequest, DenseRole, Error, ExecutionPhase,
    OutputHeadExecution, Result, backend::output::projection::OutputHeadWeight,
};

impl CudaOutputHeadTemplate {
    pub(crate) fn instantiate(&self, backend: &CudaBackend) -> Result<CudaOutputHead> {
        match &self.weight {
            OutputHeadWeight::Bf16(weight) => Ok(CudaOutputHead::bf16(
                backend.prepare_bf16_projection(DensePlanRequest {
                    phase: ExecutionPhase::Decode,
                    role: DenseRole::OutputHead,
                    tokens: 1,
                    input_features: self.input_features,
                    output_features: self.output_features,
                })?,
                weight.clone(),
                backend.inner.stream.clone(),
            )),
            OutputHeadWeight::Fp8 { weight, scales, row_scales } => {
                self.instantiate_fp8(backend, weight, scales, row_scales)
            },
            OutputHeadWeight::Fp8Residual {
                weight,
                row_scales,
                residual,
                residual_scales,
            } => Ok(CudaOutputHead::fp8_residual(
                self.kernels
                    .clone()
                    .ok_or(Error::InvalidExecutionPlan("missing residual output kernels"))?,
                weight.clone(),
                row_scales.clone(),
                residual.clone(),
                residual_scales.clone(),
                backend.inner.stream.clone(),
            )),
            OutputHeadWeight::Fp8BlockVectorized { kernels, weight, scales } => {
                Ok(CudaOutputHead::fp8_block_vectorized(
                    kernels.clone(),
                    weight.clone(),
                    scales.clone(),
                    backend.inner.stream.clone(),
                ))
            },
            OutputHeadWeight::Fp8BlockRefined {
                kernels,
                refinement,
                exact_tensor,
                exact_weight,
                weight,
                scales,
            } => match self.execution {
                OutputHeadExecution::AutoRefined => CudaOutputHead::auto_refined(
                    backend,
                    kernels.clone(),
                    refinement.clone(),
                    exact_tensor.clone(),
                    exact_weight.clone(),
                    weight.clone(),
                    scales.clone(),
                    self.input_features,
                    self.output_features,
                ),
                OutputHeadExecution::Fp8BlockRefined => CudaOutputHead::fp8_block_refined(
                    backend,
                    kernels.clone(),
                    refinement.clone(),
                    exact_weight.clone(),
                    weight.clone(),
                    scales.clone(),
                    self.output_features,
                ),
                _ => Err(Error::InvalidExecutionPlan(
                    "refined output storage differs from execution plan",
                )),
            },
        }
    }

    fn instantiate_fp8(
        &self,
        backend: &CudaBackend,
        weight: &mircuda::DeviceBuffer<u8>,
        scales: &mircuda::DeviceBuffer<f32>,
        row_scales: &mircuda::DeviceBuffer<f32>,
    ) -> Result<CudaOutputHead> {
        let kernels = self
            .kernels
            .clone()
            .ok_or(Error::InvalidExecutionPlan("missing FP8 output kernels"))?;
        match self.execution {
            OutputHeadExecution::Fp8Blockwise => CudaOutputHead::fp8(
                backend,
                kernels,
                weight.clone(),
                scales.clone(),
                row_scales.clone(),
                self.input_features,
                self.output_features,
            ),
            OutputHeadExecution::Fp8Vectorized => Ok(CudaOutputHead::fp8_vectorized(
                kernels,
                weight.clone(),
                row_scales.clone(),
                backend.inner.stream.clone(),
            )),
            _ => Err(Error::InvalidExecutionPlan(
                "per-row FP8 output storage differs from execution plan",
            )),
        }
    }
}
