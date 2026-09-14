use super::{batch::common_position, *};
use crate::engine::{QuantizedArrays, QuantizedLinear};

mod batch;
mod history;

#[test]
fn executes_gated_grouped_query_attention_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let attention = fixture(&stream)?;
    let input = Array::from_f32(&vec![0.0; 128], &[1, 2, 64])?;
    let mut cache = KvCache::new(16)?;
    let output = attention.forward(&input, &mut cache, 0, 0, true, &stream)?;

    output.async_eval(&stream)?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 2, 64]);
    assert!(output.to_vec_f32(&stream)?.iter().all(|value| *value == 0.0));
    assert_eq!(cache.offset()?, 2);
    Ok(())
}

#[test]
fn batches_only_rows_with_a_common_position() {
    assert_eq!(common_position(&[4, 4, 4]), Some(4));
    assert_eq!(common_position(&[4, 5]), None);
    assert_eq!(common_position(&[]), None);
}

fn linear(output_width: i32, stream: &Stream) -> Result<BoundLinear> {
    let weights = (0..output_width * 64)
        .map(|index| Ok(f32::from(i16::try_from((index * 13) % 23 - 11)?) * 0.01))
        .collect::<Result<Vec<_>>>()?;
    let dense = Array::from_f32(&weights, &[output_width, 64])?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}

fn fixture(stream: &Stream) -> Result<GatedFullAttention> {
    let config = GatedFullAttentionConfig {
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 64,
        rope_dimensions: 64,
        rope_base: 10_000.0,
        attention_scale: 0.125,
        rms_norm_eps: 1.0e-6,
        rope_layout: RotaryEmbeddingLayout::Standard,
    };
    Ok(GatedFullAttention {
        key_value_join: None,
        config,
        query: linear(128, stream)?,
        key: linear(64, stream)?,
        value: linear(64, stream)?,
        output: linear(64, stream)?,
        query_norm: NormWeight::from_weight(Array::from_f32(&vec![1.0; 64], &[64])?),
        key_norm: NormWeight::from_weight(Array::from_f32(&vec![1.0; 64], &[64])?),
    })
}
