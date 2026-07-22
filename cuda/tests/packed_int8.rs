use std::path::PathBuf;

use libmir_cuda::{
    Result,
    kernels::{PackedInt8Launch, PackedInt8Linear, PackedInt8Spec},
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};

#[test]
fn decodes_compressed_tensors_int8_for_gemv_and_qmm() -> Result<()> {
    for tokens in [1, 2] {
        check(tokens)?;
    }
    Ok(())
}

fn check(tokens: usize) -> Result<()> {
    let driver = mircuda::Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let token_values = &[1.0_f32, 2.0][..tokens];
    let input_values = token_values
        .iter()
        .flat_map(|&value| [bf16::from_f32(value); 16])
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &weights())?;
    let scales =
        copy_device(&context, &stream, &pool, &[bf16::from_f32(0.5), bf16::from_f32(0.25)])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, tokens * 2)?;
    let compiler =
        Compiler::with_include_paths(context.clone(), [PathBuf::from("/usr/local/cuda/include")])?;
    let operation = PackedInt8Linear::compile(&compiler, PackedInt8Spec::new(tokens, 16, 2)?)?;

    operation.execute(
        &stream,
        &mut PackedInt8Launch {
            input: &input,
            weight: &weight,
            scales: &scales,
            output: &mut output,
        },
    )?;

    let mut host = context.allocate_pinned::<bf16>(tokens * 2)?;
    stream.copy_to_host(&output, &mut host)?;
    let expected = token_values
        .iter()
        .flat_map(|&value| [bf16::from_f32(-4.0 * value), bf16::from_f32(8.0 * value)])
        .collect::<Vec<_>>();
    assert_eq!(host.to_vec()?, expected);
    Ok(())
}

fn weights() -> Vec<i32> {
    let first = pack([-2, -1, 0, 1]);
    let second = pack([2, 2, 2, 2]);
    vec![first; 4].into_iter().chain(vec![second; 4]).collect()
}

fn pack(values: [i8; 4]) -> i32 {
    i32::from_le_bytes(values.map(|value| value.to_ne_bytes()[0].wrapping_add(128)))
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
