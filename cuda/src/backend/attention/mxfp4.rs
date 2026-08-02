use crate::{CudaBackend, DecodeQkvWeights, Error, MxFp4Bf16Linear, Result};

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
