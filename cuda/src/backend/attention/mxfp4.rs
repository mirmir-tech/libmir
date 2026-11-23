use mircuda::{DeviceBuffer, bf16};

use super::qkv::QkvProjectionBuffers;
use crate::{
    CudaBackend, CudaTensor, DecodeQkvWeights, Error, MxFp4Bf16Linear, Result, RmsNormBf16,
};

pub(super) fn prepare_qkv(
    backend: &CudaBackend,
    tokens: usize,
    weights: Option<DecodeQkvWeights<'_>>,
) -> Result<Box<[MxFp4Bf16Linear; 3]>> {
    let DecodeQkvWeights::MxFp4(weights) =
        weights.ok_or(Error::InvalidExecutionPlan("MXFP4 attention requires prepared QKV"))?
    else {
        return Err(Error::InvalidExecutionPlan("MXFP4 attention received other QKV weights"));
    };
    Ok(Box::new([
        weights[0].prepare(backend, tokens)?,
        weights[1].prepare(backend, tokens)?,
        weights[2].prepare(backend, tokens)?,
    ]))
}

pub(super) fn execute_qkv(
    operations: &[MxFp4Bf16Linear; 3],
    input: &DeviceBuffer<bf16>,
    input_norm: &RmsNormBf16,
    norm_weight: Option<&CudaTensor>,
    weights: DecodeQkvWeights<'_>,
    buffers: &mut QkvProjectionBuffers<'_>,
) -> Result<bool> {
    let DecodeQkvWeights::MxFp4(weights) = weights else {
        return Err(Error::InvalidExecutionPlan("MXFP4 QKV operation received other weights"));
    };
    input_norm.apply(input, norm_weight, buffers.normalized)?;
    for ((operation, weight), output) in
        operations.iter().zip(weights).zip(buffers.separate.iter_mut())
    {
        operation.execute(buffers.normalized, weight, output)?;
    }
    Ok(true)
}
