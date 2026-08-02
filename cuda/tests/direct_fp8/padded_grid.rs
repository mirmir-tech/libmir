use libmir_cuda::{
    Result,
    kernels::{
        DirectFp8Activation, DirectFp8Linear, DirectFp8Scale, DirectFp8Scales, DirectFp8Spec,
    },
};
use mircuda::bf16;

use super::{copy_device, read, resources};

const INPUT: usize = 12;
const OUTPUT: usize = 5;

#[test]
fn executes_padded_block_grid_without_padded_weights() -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::ONE; INPUT])?;
    let weight = copy_device(&context, &stream, &pool, &[0x38_u8; INPUT * OUTPUT])?;
    let scale_values = [1.0_f32, 2.0, 3.0, 4.0];
    let scales = copy_device(&context, &stream, &pool, &scale_values)?;
    let input_scale = copy_device(&context, &stream, &pool, &[1.0_f32])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, OUTPUT)?;
    let scale = DirectFp8Scale::BlockGrid {
        output_groups: 2,
        input_groups: 2,
        output_block_size: 4,
        input_block_size: 8,
    };
    DirectFp8Linear::compile(
        &compiler,
        DirectFp8Spec::new(1, INPUT, OUTPUT, scale, false, DirectFp8Activation::Bf16)?,
    )?
    .execute(
        &stream,
        &input,
        &weight,
        DirectFp8Scales {
            weight: &scales,
            activation: &input_scale,
        },
        None,
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = [
        bf16::from_f32(16.0),
        bf16::from_f32(16.0),
        bf16::from_f32(16.0),
        bf16::from_f32(16.0),
        bf16::from_f32(40.0),
    ];
    assert_eq!(actual, expected);
    Ok(())
}
