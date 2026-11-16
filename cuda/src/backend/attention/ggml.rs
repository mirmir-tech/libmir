use mircuda::{DeviceBuffer, bf16};

use super::qkv::QkvProjectionBuffers;
use crate::{
    CudaBackend, CudaTensor, DecodeQkvWeights, DenseRole, Error, GgmlBf16Linear, Result,
    RmsNormBf16,
};

pub(super) fn prepare_qkv(
    backend: &CudaBackend,
    tokens: usize,
    weights: Option<DecodeQkvWeights<'_>>,
) -> Result<Box<[GgmlBf16Linear; 3]>> {
    let DecodeQkvWeights::Ggml(weights) =
        weights.ok_or(Error::InvalidExecutionPlan("GGML attention requires prepared QKV"))?
    else {
        return Err(Error::InvalidExecutionPlan("GGML attention received other QKV weights"));
    };
    Ok(Box::new([
        weights[0].prepare(backend, tokens, DenseRole::AttentionQkv)?,
        weights[1].prepare(backend, tokens, DenseRole::AttentionQkv)?,
        weights[2].prepare(backend, tokens, DenseRole::AttentionQkv)?,
    ]))
}

pub(super) fn execute_qkv(
    operations: &[GgmlBf16Linear; 3],
    input: &DeviceBuffer<bf16>,
    input_norm: &RmsNormBf16,
    norm_weight: &CudaTensor,
    weights: DecodeQkvWeights<'_>,
    buffers: &mut QkvProjectionBuffers<'_>,
) -> Result<bool> {
    let DecodeQkvWeights::Ggml(weights) = weights else {
        return Err(Error::InvalidExecutionPlan("GGML QKV operation received other weights"));
    };
    input_norm.execute(input, norm_weight, buffers.normalized)?;
    // One input: the first projection quantizes it, the others reuse that.
    for (index, ((operation, weight), output)) in
        operations.iter().zip(weights).zip(buffers.separate.iter_mut()).enumerate()
    {
        operation.execute_shared(buffers.normalized, weight, output, index > 0)?;
    }
    Ok(true)
}
