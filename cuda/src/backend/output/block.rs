use super::{
    projection::OutputHeadWeight,
    weight::{CudaOutputHeadTemplate, bf16},
};
use crate::{
    CudaBackend, CudaTensor, OutputHeadExecution, Result,
    kernels::{BlockFp8LinearKernels, BlockFp8LinearSpec, Fp8OutputSpec, Fp8RefinementKernels},
};

pub(super) fn prepare(
    backend: &CudaBackend,
    source: &CudaTensor,
    input_features: usize,
    output_features: usize,
    execution: OutputHeadExecution,
) -> Result<CudaOutputHeadTemplate> {
    let source_weight = bf16(source)?;
    let spec = BlockFp8LinearSpec::new(input_features, output_features)?;
    let kernels = BlockFp8LinearKernels::compile(&backend.inner.compiler, spec)?;
    let mut weight = backend
        .inner
        .pool
        .allocate::<u8>(&backend.inner.stream, spec.weight_elements()?)?;
    let mut scales = backend
        .inner
        .pool
        .allocate::<f32>(&backend.inner.stream, spec.scale_elements()?)?;
    kernels.quantize(&backend.inner.stream, source_weight, &mut weight, &mut scales)?;
    tracing::debug!(
        input_features,
        output_features,
        scale_block = 128,
        "prepared block-scaled E4M3 CUDA output head"
    );
    let weight = match execution {
        OutputHeadExecution::Fp8BlockVectorized => {
            OutputHeadWeight::Fp8BlockVectorized { kernels, weight, scales }
        },
        OutputHeadExecution::AutoRefined | OutputHeadExecution::Fp8BlockRefined => {
            tracing::debug!(
                candidates = Fp8RefinementKernels::candidate_count(),
                "prepared exact BF16 output candidate refinement"
            );
            OutputHeadWeight::Fp8BlockRefined {
                refinement: Fp8RefinementKernels::compile(
                    &backend.inner.compiler,
                    Fp8OutputSpec::new(input_features, output_features)?,
                )?,
                exact_tensor: source.clone(),
                exact_weight: source_weight.clone(),
                kernels,
                weight,
                scales,
            }
        },
        _ => {
            return Err(crate::Error::InvalidExecutionPlan(
                "non-block output execution reached block preparation",
            ));
        },
    };
    Ok(CudaOutputHeadTemplate {
        exact: source.clone(),
        weight,
        kernels: None,
        execution,
        input_features,
        output_features,
    })
}
