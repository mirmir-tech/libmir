use super::*;
use crate::engine::{QuantizedArrays, QuantizedLinear};

#[test]
fn executes_a_complete_gated_delta_layer_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let config = GatedDeltaLayerConfig::from_linear_attention(
        &LinearAttentionConfig {
            convolution_kernel_size: 2,
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 64,
            value_head_dim: 64,
        },
        1.0e-6,
    )?;
    let mut layer = GatedDeltaLayer {
        config,
        in_proj_qkv: linear(192, &stream)?,
        in_proj_z: linear(64, &stream)?,
        in_proj_b: linear(1, &stream)?,
        in_proj_a: linear(1, &stream)?,
        out_proj: linear(64, &stream)?,
        conv_weight: Array::from_f32(&vec![0.0; 384], &[192, 2, 1])?,
        norm_weight: NormWeight::from_weight(Array::from_f32(&vec![1.0; 64], &[64])?),
        a_log: Array::from_f32(&[0.0], &[1])?,
        dt_bias: Array::from_f32(&[0.0], &[1])?,
        compiled_decode: None,
    };
    layer.compiled_decode = CompiledDecode::new(&layer, &stream)?;
    let input = Array::from_f32(&vec![0.0; 128], &[1, 2, 64])?;
    let mut state = GatedDeltaState::new()?;
    let output = layer.forward(&input, &mut state, &stream)?;

    output.async_eval(&stream)?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 2, 64]);
    assert!(output.to_vec_f32(&stream)?.iter().all(|value| *value == 0.0));
    assert_eq!(state.offset()?, 2);
    let decode = Array::from_f32(&vec![0.0; 64], &[1, 1, 64])?;
    let output = layer.forward(&decode, &mut state, &stream)?;

    output.async_eval(&stream)?;
    stream.synchronize()?;
    assert!(output.to_vec_f32(&stream)?.iter().all(|value| *value == 0.0));
    assert_eq!(state.offset()?, 3);
    Ok(())
}

fn linear(output_width: i32, stream: &Stream) -> Result<BoundLinear> {
    let values = vec![0.0; usize::try_from(output_width * 64)?];
    let dense = Array::from_f32(&values, &[output_width, 64])?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}
