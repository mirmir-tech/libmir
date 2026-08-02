use mircuda::{DeviceBuffer, bf16};

use super::GateUpBuffers;
use crate::{
    CudaBackend, CudaTensor, DenseGateUpWeights, Error, MxFp8Bf16Linear, Result, RmsNormBf16,
};

pub(super) fn prepare_gate_up(
    backend: &CudaBackend,
    tokens: usize,
    weights: Option<DenseGateUpWeights<'_>>,
) -> Result<Box<[MxFp8Bf16Linear; 2]>> {
    let DenseGateUpWeights::MxFp8 { gate, up } =
        weights.ok_or(Error::InvalidExecutionPlan("MXFP8 MLP requires gate/up weights"))?
    else {
        return Err(Error::InvalidExecutionPlan("MXFP8 MLP received other gate/up weights"));
    };
    Ok(Box::new([gate.prepare(backend, tokens)?, up.prepare(backend, tokens)?]))
}

pub(super) fn execute_gate_up(
    operations: &[MxFp8Bf16Linear; 2],
    input: &DeviceBuffer<bf16>,
    input_norm: &RmsNormBf16,
    norm_weight: &CudaTensor,
    weights: DenseGateUpWeights<'_>,
    buffers: &mut GateUpBuffers<'_>,
) -> Result<bool> {
    let DenseGateUpWeights::MxFp8 { gate, up } = weights else {
        return Err(Error::InvalidExecutionPlan("MXFP8 gate/up operation received other weights"));
    };
    input_norm.execute(input, norm_weight, buffers.normalized)?;
    for ((operation, weight), output) in
        operations.iter().zip([gate, up]).zip(buffers.separate.iter_mut())
    {
        operation.execute(buffers.normalized, weight, output)?;
    }
    Ok(true)
}
