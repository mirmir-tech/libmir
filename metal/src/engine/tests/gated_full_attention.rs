use super::*;
use crate::engine::QuantizedArrays;

#[test]
fn executes_gated_grouped_query_attention_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
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
    let attention = GatedFullAttention {
        config,
        query: linear(128, &stream)?,
        key: linear(64, &stream)?,
        value: linear(64, &stream)?,
        output: linear(64, &stream)?,
        query_norm: NormWeight::from_weight(Array::from_f32(&vec![1.0; 64], &[64])?),
        key_norm: NormWeight::from_weight(Array::from_f32(&vec![1.0; 64], &[64])?),
    };
    let input = Array::from_f32(&vec![0.0; 128], &[1, 2, 64])?;
    let mut cache = KvCache::new(16)?;
    let output = attention.forward(&input, &mut cache, 0, 0, true, &stream)?;

    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 2, 64]);
    assert!(output.to_vec_f32()?.iter().all(|value| *value == 0.0));
    assert_eq!(cache.offset()?, 2);
    Ok(())
}

fn linear(output_width: i32, stream: &Stream) -> Result<QuantizedLinear> {
    let dense =
        Array::from_f32(&vec![0.0; usize::try_from(output_width * 64)?], &[output_width, 64])?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(QuantizedLinear::from_quantized(arrays, 64, 4))
}
