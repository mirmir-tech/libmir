use super::{Array, DenseLinear, LayerNorm, Result, Stream};

#[test]
fn executes_dense_linear_with_optional_bias() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0, 2.0], &[1, 2])?;
    let weight = Array::from_f32(&[2.0, 0.0, 0.0, 3.0], &[2, 2])?;
    let bias = Array::from_f32(&[0.5, -0.5], &[2])?;
    let linear = DenseLinear::from_arrays(&weight, Some(bias), &stream)?;
    let output = linear.forward(&input, &stream)?;

    assert_eq!(output.to_vec_f32_on_stream(&stream)?, vec![2.5, 5.5]);
    Ok(())
}

#[test]
fn executes_layer_norm_and_plain_gelu() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0, 3.0], &[1, 2])?;
    let weight = Array::from_f32(&[2.0, 3.0], &[2])?;
    let bias = Array::from_f32(&[0.5, -0.5], &[2])?;
    let norm = LayerNorm::from_arrays(weight, bias, 1.0e-5);
    let normalized = norm.forward(&input, &stream)?.to_vec_f32_on_stream(&stream)?;
    assert!((normalized[0] + 1.499_99).abs() < 1.0e-4);
    assert!((normalized[1] - 2.499_98).abs() < 1.0e-4);

    let gelu = input.gelu_tanh(&stream)?.to_vec_f32_on_stream(&stream)?;
    assert!((gelu[0] - 0.841_192).abs() < 1.0e-5);
    assert!((gelu[1] - 2.996_363_6).abs() < 1.0e-5);

    let exact = input.gelu(&stream)?.to_vec_f32_on_stream(&stream)?;
    assert!((exact[0] - 0.841_344_7).abs() < 1.0e-5);
    assert!((exact[1] - 2.995_950_2).abs() < 1.0e-5);
    Ok(())
}

#[test]
fn applies_checkpoint_clipping_around_a_dense_projection() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[-2.0, 2.0], &[1, 2])?;
    let weight = Array::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])?;
    let linear = DenseLinear::from_clipped_arrays(
        &weight,
        (Array::from_f32(&[-1.0], &[])?, Array::from_f32(&[1.0], &[])?),
        (Array::from_f32(&[-0.5], &[])?, Array::from_f32(&[0.5], &[])?),
        &stream,
    )?;

    assert_eq!(linear.forward(&input, &stream)?.to_vec_f32_on_stream(&stream)?, vec![-0.5, 0.5]);
    Ok(())
}
