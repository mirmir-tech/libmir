use mircuda::{DeviceBuffer, bf16};

use super::GateUpBuffers;
use crate::{
    CudaBackend, CudaTensor, DenseGateUpWeights, DenseSwiGluConfig, Error, PackedIntegerBf16Linear,
    Result, RmsNormBf16,
};

pub(super) fn prepare_gate_up(
    backend: &CudaBackend,
    config: DenseSwiGluConfig,
    tokens: usize,
    weights: Option<DenseGateUpWeights<'_>>,
) -> Result<Box<[PackedIntegerBf16Linear; 2]>> {
    let DenseGateUpWeights::PackedInteger { gate, up } = weights.ok_or(
        Error::InvalidExecutionPlan("packed integer MLP requires prepared gate/up weights"),
    )?
    else {
        return Err(Error::InvalidExecutionPlan(
            "packed integer MLP received other gate/up weights",
        ));
    };
    let hidden = config.attention.hidden_size;
    let intermediate = config.intermediate_size;
    Ok(Box::new([
        PackedIntegerBf16Linear::new(backend, tokens, hidden, intermediate, gate)?,
        PackedIntegerBf16Linear::new(backend, tokens, hidden, intermediate, up)?,
    ]))
}

pub(super) fn execute_gate_up(
    operations: &[PackedIntegerBf16Linear; 2],
    input: &DeviceBuffer<bf16>,
    input_norm: &RmsNormBf16,
    norm_weight: &CudaTensor,
    weights: DenseGateUpWeights<'_>,
    buffers: &mut GateUpBuffers<'_>,
) -> Result<bool> {
    let DenseGateUpWeights::PackedInteger { gate, up } = weights else {
        return Err(Error::InvalidExecutionPlan(
            "packed integer MLP operation received other weights",
        ));
    };
    input_norm.execute(input, norm_weight, buffers.normalized)?;
    for ((operation, weight), output) in
        operations.iter().zip([gate, up]).zip(buffers.separate.iter_mut())
    {
        operation.execute(buffers.normalized, weight, output)?;
    }
    Ok(true)
}
