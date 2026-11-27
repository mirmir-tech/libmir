//! Separate query, key and value projections of the integer formats.

use super::DecodeAttentionConfig;
use crate::{
    AffineQuantizedWeight, CudaBackend, PackedIntegerBf16Linear, PackedIntegerWeight, Result,
    backend::linear::AffineProjection,
};

pub(super) fn affine(
    backend: &CudaBackend,
    tokens: usize,
    input: usize,
    output: usize,
    weight: &AffineQuantizedWeight,
) -> Result<AffineProjection> {
    let config = weight.infer_config(1, input, output)?;
    AffineProjection::new(backend, tokens, input, output, config.group_size, config.bits, weight)
}

pub(super) fn packed_integer(
    backend: &CudaBackend,
    config: DecodeAttentionConfig,
    tokens: usize,
    weights: [&PackedIntegerWeight; 3],
) -> Result<Box<[PackedIntegerBf16Linear; 3]>> {
    let head = config.cache.key_head_dim;
    let widths = [
        config.query_heads * head,
        config.cache.kv_heads * head,
        config.cache.kv_heads * config.cache.value_head_dim,
    ];
    let linear = |(output, weight)| {
        PackedIntegerBf16Linear::new(backend, tokens, config.hidden_size, output, weight)
    };
    let [query, key, value] = [0, 1, 2].map(|index| linear((widths[index], weights[index])));
    Ok(Box::new([query?, key?, value?]))
}
