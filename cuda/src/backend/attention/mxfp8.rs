use crate::{CudaBackend, DecodeQkvWeights, Error, MxFp8Bf16Linear, Result};

pub(super) fn prepare_qkv(
    backend: &CudaBackend,
    tokens: usize,
    weights: Option<DecodeQkvWeights<'_>>,
) -> Result<Box<[MxFp8Bf16Linear; 3]>> {
    let DecodeQkvWeights::MxFp8(weights) =
        weights.ok_or(Error::InvalidExecutionPlan("MXFP8 attention requires prepared QKV"))?
    else {
        return Err(Error::InvalidExecutionPlan("MXFP8 attention received other QKV weights"));
    };
    Ok(Box::new([
        weights[0].prepare(backend, tokens)?,
        weights[1].prepare(backend, tokens)?,
        weights[2].prepare(backend, tokens)?,
    ]))
}
