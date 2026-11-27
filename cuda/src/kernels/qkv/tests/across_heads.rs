//! Query and key norms over a token's whole projection (`OLMo` 2).

use super::{ROTARY, Runtime, batched, fused, rope, test_values};
use crate::{
    Result,
    kernels::{QkvNormalization, QkvPostprocessSpec, RmsNorm, Rope},
};

const SPEC: QkvPostprocessSpec = QkvPostprocessSpec {
    tokens: 2,
    query_heads: 2,
    kv_heads: 2,
    head_dim: 8,
    value_head_dim: 8,
    rotary: Some(ROTARY),
    epsilon: 1.0e-6,
    normalization: QkvNormalization::QUERY_KEY_ACROSS_HEADS,
    temperature: None,
};

#[test]
fn norms_across_heads_match_a_norm_of_the_whole_projection() -> Result<()> {
    let runtime = Runtime::new()?;
    let packed = test_values(96, 0)?;
    let (query_weight, key_weight) = (test_values(16, 2)?, test_values(16, 5)?);
    let actual = fused(&runtime, SPEC, &packed, &query_weight, &key_weight, 3)?;

    let row = |token: usize, offset: usize| &packed[token * 48 + offset..token * 48 + offset + 16];
    let query: Vec<_> = (0..2).flat_map(|token| row(token, 0).to_vec()).collect();
    let key: Vec<_> = (0..2).flat_map(|token| row(token, 16).to_vec()).collect();
    let value: Vec<_> = (0..2).flat_map(|token| row(token, 32).to_vec()).collect();
    let expected_query = normalized_rope(&runtime, &query, &query_weight)?;
    let expected_key = normalized_rope(&runtime, &key, &key_weight)?;
    assert_eq!(actual.0, expected_query);
    assert_eq!(actual.1, expected_key);
    assert_eq!(actual.2, value);

    let batch = batched(&runtime, SPEC, &packed, &query_weight, &key_weight)?;
    assert_eq!(batch, actual);
    let per_head = QkvPostprocessSpec {
        normalization: QkvNormalization::QUERY_KEY,
        temperature: None,
        ..SPEC
    };
    let per_head = fused(&runtime, per_head, &packed, &query_weight, &key_weight, 3)?;
    assert_ne!(per_head.0, actual.0);
    Ok(())
}

/// Two tokens of two heads, normalized over both heads, then rotated from
/// position 3.
fn normalized_rope(
    runtime: &Runtime,
    values: &[mircuda::bf16],
    weight: &[mircuda::bf16],
) -> Result<Vec<mircuda::bf16>> {
    let input = runtime.copy(values)?;
    let weight = runtime.copy(weight)?;
    let mut normalized = runtime.allocate(values.len())?;
    RmsNorm::compile(&runtime.compiler, 2, 16, SPEC.epsilon)?
        .execute(&runtime.stream, &input, &weight, &mut normalized)?;
    let mut output = runtime.allocate(values.len())?;
    Rope::compile(&runtime.compiler, rope(SPEC, 2))?
        .execute(&runtime.stream, &normalized, &mut output, 3)?;
    runtime.read(&output)
}
