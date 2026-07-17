use mircuda::{BlockwiseFp8VectorPlan, BlockwiseFp8VectorSpec, DeviceBuffer, Stream, bf16};

mod execution;

use crate::{
    Bf16Projection, CudaBackend, CudaTensor, DensePlanRequest, DenseRole, ExecutionPhase, Result,
    kernels::{BlockFp8LinearKernels, Fp8OutputKernels, Fp8RefinementKernels},
};

#[derive(Clone, Debug)]
pub(super) enum OutputHeadWeight {
    Bf16(CudaTensor),
    Fp8 {
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
        row_scales: DeviceBuffer<f32>,
    },
    Fp8Residual {
        weight: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
        residual: DeviceBuffer<u8>,
        residual_scales: DeviceBuffer<f32>,
    },
    Fp8BlockVectorized {
        kernels: BlockFp8LinearKernels,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
    },
    Fp8BlockRefined {
        kernels: BlockFp8LinearKernels,
        refinement: Fp8RefinementKernels,
        exact_tensor: CudaTensor,
        exact_weight: DeviceBuffer<bf16>,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
    },
}

#[derive(Debug)]
enum Projection {
    Bf16 {
        operation: Bf16Projection,
        weight: CudaTensor,
    },
    Fp8 {
        operation: BlockwiseFp8VectorPlan,
        kernels: Fp8OutputKernels,
        input: DeviceBuffer<u8>,
        input_scales: DeviceBuffer<f32>,
        weight: DeviceBuffer<u8>,
        weight_scales: DeviceBuffer<f32>,
        row_scales: DeviceBuffer<f32>,
    },
    Fp8Vectorized {
        kernels: Fp8OutputKernels,
        weight: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
    },
    Fp8Residual {
        kernels: Fp8OutputKernels,
        weight: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
        residual: DeviceBuffer<u8>,
        residual_scales: DeviceBuffer<f32>,
    },
    Fp8BlockVectorized {
        kernels: BlockFp8LinearKernels,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
    },
    Fp8BlockRefined {
        kernels: BlockFp8LinearKernels,
        refinement: Fp8RefinementKernels,
        exact_weight: DeviceBuffer<bf16>,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
        first: DeviceBuffer<u64>,
        second: DeviceBuffer<u64>,
    },
    AutoRefined {
        exact_operation: Bf16Projection,
        exact_tensor: CudaTensor,
        kernels: BlockFp8LinearKernels,
        refinement: Fp8RefinementKernels,
        exact_weight: DeviceBuffer<bf16>,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
        first: DeviceBuffer<u64>,
        second: DeviceBuffer<u64>,
    },
}

/// Session-local operation over model-shared output-head storage.
#[derive(Debug)]
pub struct CudaOutputHead {
    projection: Projection,
    stream: Stream,
}

impl CudaOutputHead {
    pub(super) const fn bf16(
        operation: Bf16Projection,
        weight: CudaTensor,
        stream: Stream,
    ) -> Self {
        Self {
            projection: Projection::Bf16 { operation, weight },
            stream,
        }
    }

    pub(super) fn fp8(
        backend: &CudaBackend,
        kernels: Fp8OutputKernels,
        weight: DeviceBuffer<u8>,
        weight_scales: DeviceBuffer<f32>,
        row_scales: DeviceBuffer<f32>,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let spec = BlockwiseFp8VectorSpec::new(output_features, input_features)?;
        Ok(Self {
            projection: Projection::Fp8 {
                operation: BlockwiseFp8VectorPlan::new(
                    &backend.inner.context,
                    &backend.inner.stream,
                    spec,
                )?,
                kernels,
                input: backend.inner.pool.allocate::<u8>(&backend.inner.stream, input_features)?,
                input_scales: backend
                    .inner
                    .pool
                    .allocate::<f32>(&backend.inner.stream, spec.input_scale_elements())?,
                weight,
                weight_scales,
                row_scales,
            },
            stream: backend.inner.stream.clone(),
        })
    }

    pub(super) const fn fp8_vectorized(
        kernels: Fp8OutputKernels,
        weight: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
        stream: Stream,
    ) -> Self {
        Self {
            projection: Projection::Fp8Vectorized { kernels, weight, row_scales },
            stream,
        }
    }

    pub(super) const fn fp8_residual(
        kernels: Fp8OutputKernels,
        weight: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
        residual: DeviceBuffer<u8>,
        residual_scales: DeviceBuffer<f32>,
        stream: Stream,
    ) -> Self {
        Self {
            projection: Projection::Fp8Residual {
                kernels,
                weight,
                row_scales,
                residual,
                residual_scales,
            },
            stream,
        }
    }

    pub(super) const fn fp8_block_vectorized(
        kernels: BlockFp8LinearKernels,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
        stream: Stream,
    ) -> Self {
        Self {
            projection: Projection::Fp8BlockVectorized { kernels, weight, scales },
            stream,
        }
    }

    pub(super) fn fp8_block_refined(
        backend: &CudaBackend,
        kernels: BlockFp8LinearKernels,
        refinement: Fp8RefinementKernels,
        exact_weight: DeviceBuffer<bf16>,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
        output_features: usize,
    ) -> Result<Self> {
        let workspace = Fp8RefinementKernels::workspace_elements(output_features)?;
        Ok(Self {
            projection: Projection::Fp8BlockRefined {
                kernels,
                refinement,
                exact_weight,
                weight,
                scales,
                first: backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?,
                second: backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?,
            },
            stream: backend.inner.stream.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn auto_refined(
        backend: &CudaBackend,
        kernels: BlockFp8LinearKernels,
        refinement: Fp8RefinementKernels,
        exact_tensor: CudaTensor,
        exact_weight: DeviceBuffer<bf16>,
        weight: DeviceBuffer<u8>,
        scales: DeviceBuffer<f32>,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let workspace = Fp8RefinementKernels::workspace_elements(output_features)?;
        let exact_operation = backend.prepare_bf16_projection(DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role: DenseRole::OutputHead,
            tokens: 1,
            input_features,
            output_features,
        })?;
        Ok(Self {
            projection: Projection::AutoRefined {
                exact_operation,
                exact_tensor,
                kernels,
                refinement,
                exact_weight,
                weight,
                scales,
                first: backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?,
                second: backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?,
            },
            stream: backend.inner.stream.clone(),
        })
    }
}
