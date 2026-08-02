use libmir_cuda::kernels::{
    DirectFp8Embedding, DirectFp8EmbeddingBatch, DirectFp8EmbeddingSpec, DirectFp8Format,
    DirectFp8Scale,
};

use super::*;

#[test]
fn selects_e5m2_rows_with_compact_f32_and_bf16_scales() -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let weight = copy_device(&context, &stream, &pool, &weight_bytes_e5m2())?;
    let selected = copy_device(&context, &stream, &pool, &[1_u32, 0])?;
    let grid = DirectFp8Scale::BlockGrid {
        output_groups: 2,
        input_groups: 2,
        output_block_size: 1,
        input_block_size: 4,
    };
    let scales = copy_device(&context, &stream, &pool, &[0.5_f32, 1.0, 0.25, 0.5])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 2 * INPUT)?;
    let batch = DirectFp8EmbeddingBatch {
        selected: &selected,
        selected_start: 0,
        tokens: 2,
    };
    DirectFp8Embedding::compile(
        &compiler,
        DirectFp8EmbeddingSpec {
            format: DirectFp8Format::E5M2,
            vocab: OUTPUT,
            hidden: INPUT,
            scale: grid,
            inverse_scale: false,
            output_scale: 2.0,
        },
    )?
    .execute_f32_scales(&stream, &weight, &scales, batch, &mut output)?;
    assert_eq!(
        read(&context, &stream, &output)?,
        [1.0, -0.5, 0.25, 0.0, 1.0, -2.0, 0.5, -0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,]
            .map(bf16::from_f32)
    );

    let scales = [bf16::from_f32(0.5), bf16::from_f32(0.25)];
    let scales = copy_device(&context, &stream, &pool, &scales)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, INPUT)?;
    let batch = DirectFp8EmbeddingBatch {
        selected: &selected,
        selected_start: 1,
        tokens: 1,
    };
    DirectFp8Embedding::compile(
        &compiler,
        DirectFp8EmbeddingSpec {
            format: DirectFp8Format::E5M2,
            vocab: OUTPUT,
            hidden: INPUT,
            scale: DirectFp8Scale::OutputChannel,
            inverse_scale: true,
            output_scale: 1.0,
        },
    )?
    .execute_bf16_scales(&stream, &weight, &scales, batch, &mut output)?;
    assert_eq!(read(&context, &stream, &output)?, [bf16::from_f32(2.0); INPUT]);
    Ok(())
}
