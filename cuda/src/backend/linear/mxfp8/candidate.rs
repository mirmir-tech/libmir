use mircuda::{DeviceBuffer, MemoryPool, MxFp8Matmul, MxFp8Spec, MxFp8TensorCore, Stream, bf16};

use super::{MxFp8CheckpointWeight, dtype};
use crate::{CudaBackend, Error, Result, backend::tuning::MxFp8ProjectionExecution};

#[derive(Debug)]
pub(super) struct Candidate {
    pub(super) execution: MxFp8ProjectionExecution,
    operation: Operation,
}

#[derive(Debug)]
enum Operation {
    Portable(MxFp8Matmul),
    TensorCore(MxFp8TensorCore),
}

impl Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        spec: MxFp8Spec,
        weight: &MxFp8CheckpointWeight,
        execution: MxFp8ProjectionExecution,
    ) -> Result<Self> {
        if execution == MxFp8ProjectionExecution::TensorCore && !tensor_core_admitted(backend, spec)
        {
            return Err(Error::InvalidExecutionPlan("MXFP8 Tensor Core candidate is unavailable"));
        }
        let operation = match execution {
            MxFp8ProjectionExecution::Portable => {
                Operation::Portable(MxFp8Matmul::compile(&backend.inner.compiler, spec)?)
            },
            MxFp8ProjectionExecution::TensorCore => {
                let operation = MxFp8TensorCore::new_with_scratch(
                    &backend.inner.compiler,
                    &backend.inner.context,
                    &backend.inner.stream,
                    spec,
                    backend.mxfp8_tensor_core_scratch(spec)?,
                )?;
                let scales = tensor_core_scales(
                    &operation,
                    &backend.inner.pool,
                    &backend.inner.stream,
                    weight,
                )?;
                operation.prepare_weight_scales(&backend.inner.stream, scales)?;
                Operation::TensorCore(operation)
            },
        };
        Ok(Self { execution, operation })
    }

    pub(super) fn execute(
        &self,
        stream: &Stream,
        pool: &MemoryPool,
        input: &DeviceBuffer<bf16>,
        weight: &MxFp8CheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let packed = weight.weight.as_u32().ok_or_else(|| dtype(&weight.weight, "U32"))?;
        match &self.operation {
            Operation::Portable(operation) => Ok(operation.execute(
                stream,
                input,
                packed,
                weight.scales.as_u8().ok_or_else(|| dtype(&weight.scales, "U8"))?,
                output,
            )?),
            Operation::TensorCore(operation) => Ok(operation.execute(
                stream,
                input,
                packed,
                tensor_core_scales(operation, pool, stream, weight)?,
                output,
            )?),
        }
    }
}

pub(super) fn tensor_core_admitted(backend: &CudaBackend, spec: MxFp8Spec) -> bool {
    spec.tokens() > 1
        && spec.input_features().is_multiple_of(128)
        && backend.inner.device.compute_capability.0 == 12
}

fn tensor_core_scales<'a>(
    operation: &MxFp8TensorCore,
    pool: &MemoryPool,
    stream: &Stream,
    weight: &'a MxFp8CheckpointWeight,
) -> Result<&'a DeviceBuffer<u8>> {
    if weight.swizzled_scales.get().is_none() {
        let scales = weight.scales.as_u8().ok_or_else(|| dtype(&weight.scales, "U8"))?;
        let candidate = operation.swizzle_weight_scales(pool, stream, scales)?;
        drop(weight.swizzled_scales.set(candidate));
    }
    weight
        .swizzled_scales
        .get()
        .ok_or(Error::InvalidExecutionPlan("MXFP8 swizzled checkpoint scales are missing"))
}
