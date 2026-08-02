use std::path::PathBuf;

use libmir_cuda::{
    Result,
    kernels::{
        PackedInt8Embedding, PackedInt8EmbeddingLaunch, PackedInt8EmbeddingSpec, PackedInt8Launch,
        PackedInt8Linear, PackedInt8Spec,
    },
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};

const INPUT: usize = 16;
const GROUP: usize = 8;

#[test]
fn decodes_grouped_compressed_tensors_int4_for_gemv_and_qmm() -> Result<()> {
    for tokens in [1, 2] {
        check(tokens)?;
    }
    Ok(())
}

#[test]
fn gathers_grouped_compressed_tensors_int4_embeddings() -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let selected = copy_device(&context, &stream, &pool, &[1_u32, 0])?;
    let weight = copy_device(&context, &stream, &pool, &weights())?;
    let scales = copy_device(&context, &stream, &pool, &scales())?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, INPUT * 2)?;
    let operation = PackedInt8Embedding::compile(
        &compiler,
        PackedInt8EmbeddingSpec::new_packed(2, INPUT, 1.0, 4, GROUP)?,
    )?;
    operation.execute(
        &stream,
        &mut PackedInt8EmbeddingLaunch {
            selected: &selected,
            selected_start: 0,
            tokens: 2,
            weight: &weight,
            scales: &scales,
            output: &mut output,
        },
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = (0..2)
        .flat_map(|selected| {
            let row = 1 - selected;
            (0..INPUT).map(move |feature| {
                bf16::from_f32(f32::from(value(row, feature)) * scale(row, feature))
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

fn check(tokens: usize) -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input_values = (0..tokens)
        .flat_map(|token| {
            (0..INPUT).map(move |feature| bf16::from_f32(input_value(token, feature)))
        })
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &weights())?;
    let scales = copy_device(&context, &stream, &pool, &scales())?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, tokens * 2)?;
    let operation = PackedInt8Linear::compile(
        &compiler,
        PackedInt8Spec::new_packed(tokens, INPUT, 2, 4, GROUP)?,
    )?;
    operation.execute(
        &stream,
        &mut PackedInt8Launch {
            input: &input,
            weight: &weight,
            scales: &scales,
            output: &mut output,
        },
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = (0..tokens)
        .flat_map(|token| {
            (0..2).map(move |row| {
                let sum = (0..INPUT)
                    .map(|feature| {
                        input_value(token, feature)
                            * f32::from(value(row, feature))
                            * scale(row, feature)
                    })
                    .sum::<f32>();
                bf16::from_f32(sum)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

fn weights() -> Vec<i32> {
    (0..2)
        .flat_map(|row| {
            (0..2).map(move |word| {
                let packed = (0..8).fold(0_u32, |packed, nibble| {
                    let encoded =
                        u32::from(value(row, word * 8 + nibble).to_ne_bytes()[0].wrapping_add(8));
                    packed | (encoded << (nibble * 4))
                });
                i32::from_ne_bytes(packed.to_ne_bytes())
            })
        })
        .collect()
}

fn scales() -> Vec<bf16> {
    (0..2)
        .flat_map(|row| (0..2).map(move |group| bf16::from_f32(scale(row, group * GROUP))))
        .collect()
}

fn value(row: usize, feature: usize) -> i8 {
    if row == 0 {
        i8::try_from(feature % GROUP).unwrap_or_default() - 4
    } else if feature.is_multiple_of(2) {
        3
    } else {
        -2
    }
}

fn scale(row: usize, feature: usize) -> f32 {
    [[0.5, 0.25], [0.125, 1.0]][row][feature / GROUP]
}

fn input_value(token: usize, feature: usize) -> f32 {
    if token == 0 {
        f32::from(u16::try_from(feature + 1).unwrap_or_default())
    } else if feature.is_multiple_of(2) {
        -1.0
    } else {
        2.0
    }
}

fn resources() -> Result<(Context, Stream, MemoryPool, Compiler)> {
    let driver = mircuda::Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler =
        Compiler::with_include_paths(context.clone(), [PathBuf::from("/usr/local/cuda/include")])?;
    Ok((context, stream, pool, compiler))
}

fn copy_device<T: DeviceElement + Copy>(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate::<T>(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement + Copy>(
    context: &Context,
    stream: &Stream,
    values: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = context.allocate_pinned::<T>(values.len())?;
    stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
