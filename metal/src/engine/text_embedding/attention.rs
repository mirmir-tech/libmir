use super::{LayerConfig, weights::AttentionWeights};
use crate::engine::{Array, Result, RopeOptions, Stream};

pub(super) fn forward(
    input: &Array,
    weights: &AttentionWeights,
    config: LayerConfig,
    stream: &Stream,
) -> Result<Array> {
    let sequence = input.shape()?.get(1).copied().ok_or_else(|| {
        crate::engine::Error::InvalidModel("embedding attention input has no sequence axis".into())
    })?;
    let queries = weights
        .query
        .forward(input, stream)?
        .reshape(&[1, sequence, config.heads, config.head_dim], stream)?;
    let keys = weights
        .key
        .forward(input, stream)?
        .reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?;
    let values = weights
        .value
        .forward(input, stream)?
        .reshape(&[1, sequence, config.kv_heads, config.head_dim], stream)?;
    let queries = normalize(queries, weights.query_norm.as_ref(), config, stream)?;
    let keys = normalize(keys, weights.key_norm.as_ref(), config, stream)?;
    let queries = rotate(&queries, config, stream)?;
    let keys = rotate(&keys, config, stream)?;
    let values = values.transpose(&[0, 2, 1, 3], stream)?;
    let output = queries.scaled_dot_product_attention(
        &keys,
        &values,
        config.attention_scale,
        true,
        stream,
    )?;
    weights.output.forward(
        &output
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[1, sequence, config.query_width], stream)?,
        stream,
    )
}

fn normalize(
    input: Array,
    weight: Option<&crate::engine::NormWeight>,
    config: LayerConfig,
    stream: &Stream,
) -> Result<Array> {
    match weight {
        Some(weight) => weight.apply(&input, config.rms_norm_eps, stream),
        None => Ok(input),
    }
}

fn rotate(input: &Array, config: LayerConfig, stream: &Stream) -> Result<Array> {
    input.transpose(&[0, 2, 1, 3], stream)?.rope(
        RopeOptions {
            dimensions: config.head_dim,
            traditional: false,
            base: Some(config.rope_base),
            scale: 1.0,
            offset: 0,
        },
        stream,
    )
}
