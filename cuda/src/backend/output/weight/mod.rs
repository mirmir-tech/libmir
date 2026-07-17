use super::projection::OutputHeadWeight;
use crate::{
    CudaBackend, CudaTensor, Error, OutputHeadExecution, OutputHeadPlanRequest, Result,
    kernels::{Fp8OutputKernels, Fp8OutputSpec, Fp8ResidualWeightBuffers},
};

mod instantiate;

/// Model-owned output weight prepared once and shared by all sessions.
#[derive(Clone, Debug)]
pub struct CudaOutputHeadTemplate {
    pub(super) exact: CudaTensor,
    pub(super) weight: OutputHeadWeight,
    pub(super) kernels: Option<Fp8OutputKernels>,
    pub(super) execution: OutputHeadExecution,
    pub(super) input_features: usize,
    pub(super) output_features: usize,
}

impl CudaOutputHeadTemplate {
    pub(crate) fn prepare(
        backend: &CudaBackend,
        source: CudaTensor,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let plan = backend
            .execution_planner()
            .plan_output_head(OutputHeadPlanRequest { input_features, output_features })?;
        tracing::debug!(
            input_features,
            output_features,
            execution = ?plan.execution(),
            source = ?plan.source(),
            "planned CUDA output head"
        );
        match plan.execution() {
            OutputHeadExecution::Bf16 => Ok(Self {
                exact: source.clone(),
                weight: OutputHeadWeight::Bf16(source),
                kernels: None,
                execution: OutputHeadExecution::Bf16,
                input_features,
                output_features,
            }),
            execution
            @ (OutputHeadExecution::Fp8Blockwise | OutputHeadExecution::Fp8Vectorized) => {
                Self::prepare_fp8(backend, &source, input_features, output_features, execution)
            },
            OutputHeadExecution::Fp8Residual => {
                Self::prepare_residual(backend, &source, input_features, output_features)
            },
            execution @ (OutputHeadExecution::AutoRefined
            | OutputHeadExecution::Fp8BlockVectorized
            | OutputHeadExecution::Fp8BlockRefined) => {
                super::block::prepare(backend, &source, input_features, output_features, execution)
            },
        }
    }

    fn prepare_fp8(
        backend: &CudaBackend,
        source: &CudaTensor,
        input_features: usize,
        output_features: usize,
        execution: OutputHeadExecution,
    ) -> Result<Self> {
        let source_weight = bf16(source)?;
        let spec = Fp8OutputSpec::new(input_features, output_features)?;
        let kernels = Fp8OutputKernels::compile(&backend.inner.compiler, spec)?;
        let mut weight = backend
            .inner
            .pool
            .allocate::<u8>(&backend.inner.stream, spec.weight_elements()?)?;
        let mut scales = backend
            .inner
            .pool
            .allocate::<f32>(&backend.inner.stream, spec.weight_scale_elements()?)?;
        let mut row_scales =
            backend.inner.pool.allocate::<f32>(&backend.inner.stream, output_features)?;
        kernels.quantize_weight(
            &backend.inner.stream,
            source_weight,
            &mut weight,
            &mut scales,
            &mut row_scales,
        )?;
        tracing::debug!(input_features, output_features, "prepared blockwise E4M3 output head");
        Ok(Self {
            exact: source.clone(),
            weight: OutputHeadWeight::Fp8 { weight, scales, row_scales },
            kernels: Some(kernels),
            execution,
            input_features,
            output_features,
        })
    }

    fn prepare_residual(
        backend: &CudaBackend,
        source: &CudaTensor,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let source_weight = bf16(source)?;
        let spec = Fp8OutputSpec::new(input_features, output_features)?;
        let kernels = Fp8OutputKernels::compile(&backend.inner.compiler, spec)?;
        let mut weight = backend
            .inner
            .pool
            .allocate::<u8>(&backend.inner.stream, spec.weight_elements()?)?;
        let mut block_scales = backend
            .inner
            .pool
            .allocate::<f32>(&backend.inner.stream, spec.weight_scale_elements()?)?;
        let mut row_scales =
            backend.inner.pool.allocate::<f32>(&backend.inner.stream, output_features)?;
        let mut residual = backend
            .inner
            .pool
            .allocate::<u8>(&backend.inner.stream, spec.residual_elements()?)?;
        let mut residual_scales = backend
            .inner
            .pool
            .allocate::<f32>(&backend.inner.stream, spec.residual_scale_elements()?)?;
        let mut buffers = Fp8ResidualWeightBuffers::new(
            &mut weight,
            &mut block_scales,
            &mut row_scales,
            &mut residual,
            &mut residual_scales,
        );
        kernels.quantize_weight_residual(&backend.inner.stream, source_weight, &mut buffers)?;
        tracing::debug!(input_features, output_features, "prepared FP8 plus INT4 output head");
        Ok(Self {
            exact: source.clone(),
            weight: OutputHeadWeight::Fp8Residual {
                weight,
                row_scales,
                residual,
                residual_scales,
            },
            kernels: Some(kernels),
            execution: OutputHeadExecution::Fp8Residual,
            input_features,
            output_features,
        })
    }
}

pub(super) fn bf16(source: &CudaTensor) -> Result<&mircuda::DeviceBuffer<mircuda::bf16>> {
    source.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: source.name().into(),
        expected: "BF16",
    })
}
