use crate::engine::{Array, Result, Stream};

fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).unwrap_or(0)) / 32_768.0 - 1.0
        })
        .collect()
}

fn fused(
    queries: &Array,
    keys: &Array,
    values: &Array,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Vec<f32>> {
    let mask = mask
        .map_or(mirtal::AttentionMask::None, |mask| mirtal::AttentionMask::Array(mask.native()));
    let output = Array::from_native(stream.native().graph().scaled_dot_product_attention(
        queries.native(),
        keys.native(),
        values.native(),
        mirtal::ScaledDotProductAttention { scale: 0.04, mask, sinks: None },
    )?)?;
    output.to_vec_f32(stream)
}

fn assert_close(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual) {
        assert!((expected - actual).abs() < 1e-3, "{expected} != {actual}");
    }
}

#[test]
fn grouped_head_dim_512_decode_matches_fused_attention() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let (batch, heads, kv_heads, tokens, dim) = (2, 8, 2, 37, 512);
    let queries = Array::from_f32(&values(batch * heads * dim, 1), &[2, 8, 1, 512])?;
    let keys = Array::from_f32(&values(batch * kv_heads * tokens * dim, 2), &[2, 2, 37, 512])?;
    let value_rows =
        Array::from_f32(&values(batch * kv_heads * tokens * dim, 3), &[2, 2, 37, 512])?;
    let padding = (0..batch * tokens)
        .map(|slot| {
            if slot >= tokens && slot % tokens >= 32 {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mask = Array::from_f32(&padding, &[2, 1, 1, 37])?;

    let plain = queries.scaled_dot_product_attention(&keys, &value_rows, 0.04, false, &stream)?;
    let masked =
        queries.scaled_dot_product_attention_with_mask(&keys, &value_rows, 0.04, &mask, &stream)?;

    assert!(
        queries
            .grouped_decode_attention(&keys, &value_rows, 0.04, None, &stream)?
            .is_some()
    );
    assert_close(
        &fused(&queries, &keys, &value_rows, None, &stream)?,
        &plain.to_vec_f32(&stream)?,
    );
    assert_close(
        &fused(&queries, &keys, &value_rows, Some(&mask), &stream)?,
        &masked.to_vec_f32(&stream)?,
    );
    Ok(())
}

#[test]
fn grouped_decode_leaves_other_shapes_to_mlx() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let queries = Array::from_f32(&values(2 * 256, 4), &[1, 2, 1, 256])?;
    let keys = Array::from_f32(&values(5 * 256, 5), &[1, 1, 5, 256])?;
    let prompt = Array::from_f32(&values(2 * 2 * 512, 6), &[1, 2, 2, 512])?;
    let long_keys = Array::from_f32(&values(5 * 512, 7), &[1, 1, 5, 512])?;
    let boolean =
        Array::from_f32(&[1.0; 5], &[1, 1, 1, 5])?.astype(crate::engine::Dtype::Bool, &stream)?;
    let single = Array::from_f32(&values(2 * 512, 8), &[1, 2, 1, 512])?;

    assert!(queries.grouped_decode_attention(&keys, &keys, 0.1, None, &stream)?.is_none());
    assert!(
        prompt
            .grouped_decode_attention(&long_keys, &long_keys, 0.1, None, &stream)?
            .is_none()
    );
    assert!(
        single
            .grouped_decode_attention(&long_keys, &long_keys, 0.1, Some(&boolean), &stream)?
            .is_none()
    );
    Ok(())
}
