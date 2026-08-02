use super::*;

#[test]
fn gathers_dense_expert_matrices_for_sorted_and_unsorted_routes() -> Result<()> {
    let stream = Stream::new_cpu()?;
    let weight = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2])?;
    let bias = Array::from_f32(&[10.0, 20.0], &[2, 1])?;
    let linear = DenseLinear::from_arrays(&weight, Some(bias), &stream)?;

    let indices = Array::from_u32(&[0, 1], &[1, 1, 2])?;
    let input = Array::from_f32(&[1.0, 1.0, 2.0, 1.0], &[1, 1, 2, 1, 2])?;
    assert_eq!(
        linear.gather(&input, &indices, &stream)?.to_vec_f32_on_stream(&stream)?,
        [13.0, 30.0]
    );

    let indices = Array::from_u32(&[0, 1], &[2])?;
    let input = Array::from_f32(&[1.0, 1.0, 1.0, 1.0], &[2, 1, 2])?;
    assert_eq!(
        linear.gather(&input, &indices, &stream)?.to_vec_f32_on_stream(&stream)?,
        [13.0, 27.0]
    );
    Ok(())
}
