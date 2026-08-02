use crate::{
    AffineQuantizedWeight, CudaBackend, DenseSwiGluConfig, Result,
    backend::linear::AffineProjection,
};

pub(super) fn gate_up(
    backend: &CudaBackend,
    tokens: usize,
    config: DenseSwiGluConfig,
    weight: &AffineQuantizedWeight,
) -> Result<AffineProjection> {
    let input = config.attention.hidden_size;
    let output = config.intermediate_size;
    let affine = weight.infer_config(1, input, output)?;
    AffineProjection::new(backend, tokens, input, output, affine.group_size, affine.bits, weight)
}
