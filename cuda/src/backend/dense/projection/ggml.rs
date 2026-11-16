use mircuda::{DeviceBuffer, bf16};

use super::GateUpBuffers;
use crate::{
    CudaBackend, CudaTensor, DenseGateUpWeights, DenseRole, Error, GgmlBf16Linear, Result,
    RmsNormBf16,
};

pub(super) fn prepare_gate_up(
    backend: &CudaBackend,
    tokens: usize,
    weights: Option<DenseGateUpWeights<'_>>,
) -> Result<Box<[GgmlBf16Linear; 2]>> {
    let DenseGateUpWeights::Ggml { gate, up } =
        weights.ok_or(Error::InvalidExecutionPlan("GGML MLP requires gate/up weights"))?
    else {
        return Err(Error::InvalidExecutionPlan("GGML MLP received other gate/up weights"));
    };
    Ok(Box::new([
        gate.prepare(backend, tokens, DenseRole::DenseGateUp)?,
        up.prepare(backend, tokens, DenseRole::DenseGateUp)?,
    ]))
}

pub(super) fn execute_gate_up(
    operations: &[GgmlBf16Linear; 2],
    input: &DeviceBuffer<bf16>,
    input_norm: &RmsNormBf16,
    norm_weight: &CudaTensor,
    weights: DenseGateUpWeights<'_>,
    buffers: &mut GateUpBuffers<'_>,
) -> Result<bool> {
    let DenseGateUpWeights::Ggml { gate, up } = weights else {
        return Err(Error::InvalidExecutionPlan("GGML gate/up operation received other weights"));
    };
    input_norm.execute(input, norm_weight, buffers.normalized)?;
    // One input: the first projection quantizes it, the others reuse that.
    for (index, ((operation, weight), output)) in
        operations.iter().zip([gate, up]).zip(buffers.separate.iter_mut()).enumerate()
    {
        operation.execute_shared(buffers.normalized, weight, output, index > 0)?;
    }
    Ok(true)
}
