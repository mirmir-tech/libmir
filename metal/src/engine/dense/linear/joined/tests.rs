use super::*;
use crate::engine::Dtype;

#[test]
fn joins_biased_dense_outputs_and_rejects_clipping_or_mixed_bias() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let weight =
        Array::from_f32(&vec![0.125; 17 * 64], &[17, 64])?.astype(Dtype::Bfloat16, &stream)?;
    let bias = Array::from_f32(&[0.25; 17], &[17])?.astype(Dtype::Bfloat16, &stream)?;
    let first = DenseLinear::from_arrays(&weight, Some(bias), &stream)?;
    let second_weight =
        Array::from_f32(&vec![-0.375; 31 * 64], &[31, 64])?.astype(Dtype::Bfloat16, &stream)?;
    let bias = Array::from_f32(&[0.5; 31], &[31])?.astype(Dtype::Bfloat16, &stream)?;
    let second = DenseLinear::from_arrays(&second_weight, Some(bias), &stream)?;
    let joined = first.join_outputs(&second, &stream)?.ok_or(Error::ShapeOverflow)?;
    for rows in [1_u16, 3, 5] {
        let values = (0..rows * 64).map(|i| f32::from(i % 13) / 8.0 - 0.75).collect::<Vec<_>>();
        let input = Array::from_f32(&values, &[i32::from(rows), 1, 64])?
            .astype(Dtype::Bfloat16, &stream)?;
        let (key, value) = crate::engine::fused_gate_up::split_last(
            &joined.forward(&input, &stream)?,
            17,
            &stream,
        )?;
        assert_eq!(key.to_vec_f32(&stream)?, first.forward(&input, &stream)?.to_vec_f32(&stream)?);
        assert_eq!(
            value.to_vec_f32(&stream)?,
            second.forward(&input, &stream)?.to_vec_f32(&stream)?
        );
    }
    let unbiased = DenseLinear::from_arrays(&weight, None, &stream)?;
    assert!(first.join_outputs(&unbiased, &stream)?.is_none());
    let bounds =
        || -> Result<_> { Ok((Array::from_f32(&[-1.0], &[])?, Array::from_f32(&[1.0], &[])?)) };
    let clipped = DenseLinear::from_clipped_arrays(&weight, bounds()?, bounds()?, &stream)?;
    assert!(unbiased.join_outputs(&clipped, &stream)?.is_none());
    assert!(clipped.snapshot_unclipped().is_err());
    Ok(())
}
