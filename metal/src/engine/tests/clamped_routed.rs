use super::*;
use crate::engine::kernels::MxFp4Shape;

#[test]
fn executes_native_mxfp4_clamped_experts() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let shape = MxFp4Shape {
        tokens: 1,
        top_k: 1,
        hidden: 32,
        intermediate: 32,
    };
    let input = Array::from_f32(&[1.0; 32], &[1, 32])?;
    let gate_blocks = u8_array(&vec![0x11; 64 * 16], &[1, 64, 1, 16], &stream)?;
    let gate_scales = u8_array(&vec![127; 64], &[1, 64, 1], &stream)?;
    let gate_bias = Array::from_f32(&vec![0.0; 64], &[1, 64])?;
    let down_blocks = u8_array(&vec![0x11; 32 * 16], &[1, 32, 1, 16], &stream)?;
    let down_scales = u8_array(&[127; 32], &[1, 32, 1], &stream)?;
    let down_bias = Array::from_f32(&[0.0; 32], &[1, 32])?;
    let indices = Array::from_u32(&[0], &[1, 1])?;
    let routing = Array::from_f32(&[1.0], &[1, 1])?;
    let limit = Array::from_f32(&[7.0], &[])?;

    let activated = stream
        .mxfp4_gate_up([&input, &gate_blocks, &gate_scales, &gate_bias, &indices, &limit], shape)?;
    let output = stream.mxfp4_down(
        [&activated, &down_blocks, &down_scales, &down_bias, &indices, &routing],
        shape,
    )?;
    output.async_eval()?;
    stream.synchronize()?;

    let expected_activation = 7.0 / (1.0 + (-1.702_f32 * 7.0).exp()) * 8.0;
    let expected = expected_activation * 16.0;
    assert!(output.to_vec_f32()?.iter().all(|value| (value - expected).abs() < 0.1));
    Ok(())
}

#[test]
fn executes_mlx_u32_split_mxfp4_experts() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let shape = MxFp4Shape {
        tokens: 1,
        top_k: 1,
        hidden: 32,
        intermediate: 32,
    };
    let input = Array::from_f32(&[1.0; 32], &[1, 32])?;
    let blocks = Array::from_u32(&[0x1111_1111; 32 * 4], &[1, 32, 4])?;
    let scales = u8_array(&[127; 32], &[1, 32, 1], &stream)?;
    let bias = Array::from_f32(&[0.0; 32], &[1, 32])?;
    let indices = Array::from_u32(&[0], &[1, 1])?;
    let routing = Array::from_f32(&[1.0], &[1, 1])?;
    let limit = Array::from_f32(&[7.0], &[])?;

    let activated = stream.mxfp4_split_gate_up(
        [&input, &blocks, &scales, &bias, &blocks, &scales, &bias, &indices, &limit],
        shape,
    )?;
    let output =
        stream.mxfp4_u32_down([&activated, &blocks, &scales, &bias, &indices, &routing], shape)?;
    output.async_eval()?;
    stream.synchronize()?;

    let expected_activation = 7.0 / (1.0 + (-1.702_f32 * 7.0).exp()) * 8.0;
    let expected = expected_activation * 16.0;
    assert!(output.to_vec_f32()?.iter().all(|value| (value - expected).abs() < 0.1));
    Ok(())
}

#[test]
fn learned_sink_participates_in_attention_normalization() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let query = Array::from_f32(&[0.0], &[1, 1, 1, 1])?;
    let key = Array::from_f32(&[0.0], &[1, 1, 1, 1])?;
    let value = Array::from_f32(&[2.0], &[1, 1, 1, 1])?;
    let sink = Array::from_f32(&[0.0], &[1])?;
    let output =
        query.scaled_dot_product_attention_with_sinks(&key, &value, 1.0, false, &sink, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;

    assert!((output.to_vec_f32()?[0] - 1.0).abs() < 1.0e-5);
    Ok(())
}

fn u8_array(values: &[u32], shape: &[i32], stream: &Stream) -> Result<Array> {
    Array::from_u32(values, shape)?.astype(Dtype::Uint8, stream)
}
