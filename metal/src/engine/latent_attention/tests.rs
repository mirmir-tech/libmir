//! Absorbed latent attention over the paged cache against plain attention
//! over the expanded keys and values.

use super::{
    LatentAttention,
    rotary::LatentRotary,
    row::LatentRow,
    weights::{LatentQuery, LatentUp},
};
use crate::engine::{
    Array, DenseLinear, KvCache, NormWeight, Result, Stream, binding::BoundLinear,
};

const HIDDEN: i32 = 16;
const HEADS: i32 = 2;
const NOPE: i32 = 4;
const ROPE: i32 = 4;
const RANK: i32 = 8;
const VALUE: i32 = 4;
const TOKENS: i32 = 6;

fn values(count: i32, seed: u32) -> Vec<f32> {
    (0..u32::try_from(count).unwrap_or(0))
        .map(|index| {
            let mixed = index.wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(97));
            f32::from(u16::try_from(mixed % 1000).unwrap_or(0)) / 1000.0 - 0.5
        })
        .collect()
}

fn array(shape: &[i32], seed: u32) -> Result<Array> {
    Array::from_f32(&values(shape.iter().product(), seed), shape)
}

fn linear(output: i32, input: i32, seed: u32, stream: &Stream) -> Result<BoundLinear> {
    Ok(BoundLinear::Dense(DenseLinear::from_arrays(
        &array(&[output, input], seed)?,
        None,
        stream,
    )?))
}

fn layer(stream: &Stream) -> Result<LatentAttention> {
    let wavelengths = Array::from_f32(&[1.0, 100.0], &[2])?;
    Ok(LatentAttention {
        query: LatentQuery::Direct(linear(HEADS * (NOPE + ROPE), HIDDEN, 1, stream)?),
        key_value_down: linear(RANK + ROPE, HIDDEN, 2, stream)?,
        key_value_norm: NormWeight::from_weight(Array::from_f32(&[1.0; 8], &[RANK])?),
        up: LatentUp {
            key: array(&[HEADS, NOPE, RANK], 3)?,
            value: array(&[HEADS, RANK, VALUE], 4)?,
        },
        output: linear(HIDDEN, HEADS * VALUE, 5, stream)?,
        rotary: LatentRotary {
            wavelengths,
            dimensions: ROPE,
            traditional: true,
            factor: 1.0,
            temperature: None,
        },
        row: LatentRow::new(8, 4)?,
        heads: HEADS,
        nope: NOPE,
        rope: ROPE,
        value: VALUE,
        epsilon: 1.0e-6,
        scale: 0.3,
    })
}

/// Causal attention of every token over the expanded per-head keys
/// `[W_UK · latent ‖ k_pe]` and values `W_UV · latent`.
fn expanded(layer: &LatentAttention, input: &Array, stream: &Stream) -> Result<Array> {
    let graph = stream.native().graph();
    let query = layer
        .query
        .forward(input, stream, layer.epsilon)?
        .reshape(&[1, TOKENS, HEADS, NOPE + ROPE], stream)?
        .transpose(&[0, 2, 1, 3], stream)?;
    let heads = usize::try_from(HEADS)?;
    let tokens = usize::try_from(TOKENS)?;
    let content_query = query.slice(&[0, 0, 0, 0], &[1, heads, tokens, 4], stream)?;
    let rotary_query = layer.rotary.apply(
        &query.slice(&[0, 0, 0, 4], &[1, heads, tokens, 8], stream)?,
        0,
        stream,
    )?;
    let compressed = layer.key_value_down.forward(input, stream)?.expand_dims(&[1], stream)?;
    let (latent, rotary_key) = layer.row.split(&compressed, stream)?;
    let latent = layer.key_value_norm.apply(&latent, layer.epsilon, stream)?;
    let rotary_key = layer.rotary.apply(&rotary_key, 0, stream)?;
    let content_key = latent.matmul(&layer.up.key.transpose(&[0, 2, 1], stream)?, stream)?;
    let values = latent.matmul(&layer.up.value, stream)?;
    let scores = content_query
        .matmul(&content_key.transpose(&[0, 1, 3, 2], stream)?, stream)?
        .add(
            &rotary_query.matmul(&rotary_key.transpose(&[0, 1, 3, 2], stream)?, stream)?,
            stream,
        )?
        .multiply_scalar(layer.scale, stream)?;
    let mask = (0..TOKENS)
        .flat_map(|row| {
            (0..TOKENS).map(move |column| {
                if column > row {
                    -1.0e9
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    let scores = scores.add(&Array::from_f32(&mask, &[TOKENS, TOKENS])?, stream)?;
    let probabilities = Array::from_native(graph.softmax(scores.native(), -1, true)?)?;
    let attended = probabilities
        .matmul(&values, stream)?
        .transpose(&[0, 2, 1, 3], stream)?
        .reshape(&[1, TOKENS, HEADS * VALUE], stream)?;
    layer.output.forward(&attended, stream)
}

#[test]
fn absorbed_prefill_and_decode_match_expanded_attention() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let layer = layer(&stream)?;
    let input = array(&[1, TOKENS, HIDDEN], 6)?;
    let expected = expanded(&layer, &input, &stream)?.to_vec_f32(&stream)?;
    let mut cache = KvCache::new_paged(16, 16)?;
    let hidden = usize::try_from(HIDDEN)?;
    let prefill = input.slice(&[0, 0, 0], &[1, 5, hidden], &stream)?;
    let decode = input.slice(&[0, 5, 0], &[1, 6, hidden], &stream)?;
    let mut actual = layer.forward(&prefill, &mut cache, 0, true, &stream)?.to_vec_f32(&stream)?;
    actual.extend(layer.forward(&decode, &mut cache, 5, false, &stream)?.to_vec_f32(&stream)?);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!((actual - expected).abs() <= 1.0e-4, "{index}: {actual} != {expected}");
    }
    Ok(())
}
