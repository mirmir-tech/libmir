use super::*;
use crate::engine::Dtype;

#[test]
fn joined_matrices_preserve_unequal_widths_bias_and_bf16_arithmetic() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for has_bias in [false, true] {
        let first = matrix(17, has_bias, &stream)?;
        let second = matrix(31, has_bias, &stream)?;
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
            assert_eq!(
                key.to_vec_f32(&stream)?,
                first.forward(&input, &stream)?.to_vec_f32(&stream)?
            );
            assert_eq!(
                value.to_vec_f32(&stream)?,
                second.forward(&input, &stream)?.to_vec_f32(&stream)?
            );
        }
        let mut incompatible = second.snapshot()?;
        incompatible.has_bias = !has_bias;
        assert!(first.join_outputs(&incompatible, &stream)?.is_none());
        incompatible = second.snapshot()?;
        incompatible.layout = MxFp4LinearLayout::Gathered;
        assert!(first.join_outputs(&incompatible, &stream)?.is_none());
        incompatible = second.snapshot()?;
        incompatible.input_features = 32;
        assert!(first.join_outputs(&incompatible, &stream)?.is_none());
    }
    Ok(())
}

fn matrix(rows: i32, has_bias: bool, stream: &Stream) -> Result<MxFp4Linear> {
    let output = usize::try_from(rows)?;
    let weight = Array::from_u32(
        &(0..output * 8)
            .map(|i| {
                if i % 2 == 0 {
                    0x7531_2064
                } else {
                    0xFEDA_B987
                }
            })
            .collect::<Vec<_>>(),
        &[rows, 8],
    )?;
    let scales =
        Array::from_u32(&vec![126; output * 2], &[rows, 2])?.astype(Dtype::Uint8, stream)?;
    let bias = Array::from_f32(&vec![0.125; output], &[rows])?.astype(Dtype::Bfloat16, stream)?;
    Ok(MxFp4Linear {
        weight,
        scales,
        bias,
        input_features: 64,
        output_features: output,
        has_bias,
        layout: MxFp4LinearLayout::Matrix,
    })
}
