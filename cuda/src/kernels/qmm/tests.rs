use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use super::*;

#[test]
fn matches_scalar_and_tensor_core_affine_prefill() -> Result<()> {
    for (bits, tokens) in [(4, 2), (4, 64), (8, 64)] {
        check(bits, tokens)?;
    }
    Ok(())
}

fn check(bits: usize, tokens: usize) -> Result<()> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let input_values = input_values(tokens);
    let weight_values = weights(bits);
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &weight_values)?;
    let scales =
        copy_device(&context, &stream, &pool, &[bf16::from_f32(0.5), bf16::from_f32(0.5)])?;
    let biases = copy_device(&context, &stream, &pool, &[bf16::ZERO; 2])?;
    let output_elements = tokens * 2;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, output_elements)?;
    let compiler =
        Compiler::with_include_paths(context.clone(), [PathBuf::from("/usr/local/cuda/include")])?;
    let matrix = AffineGemvSpec::new(64, 2, 64, bits)?;
    let operation = AffineQuantizedQmm::compile(&compiler, AffineQmmSpec::new(matrix, tokens)?)?;
    operation.execute(
        &stream,
        &mut AffineQmmLaunch {
            input: &input,
            weight: &weight,
            scales: &scales,
            biases: &biases,
            output: &mut output,
            matrix_index: 0,
        },
    )?;
    let mut host = context.allocate_pinned::<bf16>(output_elements)?;
    stream.copy_to_host(&output, &mut host)?;
    let expected = (0..tokens)
        .flat_map(|token| {
            let value = if token.is_multiple_of(2) {
                1.0
            } else {
                2.0
            };
            [bf16::from_f32(64.0 * value), bf16::from_f32(128.0 * value)]
        })
        .collect::<Vec<_>>();
    assert_eq!(host.to_vec()?, expected);
    Ok(())
}

fn input_values(tokens: usize) -> Vec<bf16> {
    let mut values = vec![bf16::ZERO; tokens * 64];
    for (token, chunk) in values.as_chunks_mut::<64>().0.iter_mut().enumerate() {
        let value = if token.is_multiple_of(2) {
            1.0
        } else {
            2.0
        };
        chunk.fill(bf16::from_f32(value));
    }
    values
}

fn weights(bits: usize) -> Vec<u32> {
    let values_per_word = 32 / bits;
    let words_per_row = 64 / values_per_word;
    let row_zero = (0..values_per_word).fold(0_u32, |word, lane| word | (2 << (lane * bits)));
    let row_one = (0..values_per_word).fold(0_u32, |word, lane| word | (4 << (lane * bits)));
    let mut values = vec![row_zero; words_per_row * 2];
    values[words_per_row..].fill(row_one);
    values
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
