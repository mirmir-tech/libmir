#![cfg(target_os = "linux")]

use std::path::PathBuf;

use libmir_cuda::{
    Result,
    kernels::{
        MxFp4GatheredLinear, MxFp4GatheredOperands, MxFp4GatheredSpec, MxFp4Linear, MxFp4Spec,
    },
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};

#[test]
fn executes_direct_mxfp4_projection_with_e8m0_scales() -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input = [bf16::ONE; 32].into_iter().chain([bf16::from_f32(2.0); 32]).collect::<Vec<_>>();
    let weight = [0x22_u8; 16].into_iter().chain([0x33_u8; 16]).collect::<Vec<_>>();
    let input = copy(&context, &stream, &pool, &input)?;
    let weight = copy(&context, &stream, &pool, &weight)?;
    let scales = copy(&context, &stream, &pool, &[127_u8, 128])?;
    let bias = copy(&context, &stream, &pool, &[bf16::ONE, bf16::from_f32(-2.0)])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 4)?;
    MxFp4Linear::compile(&compiler, MxFp4Spec::new(2, 32, 2)?)?.execute(
        &stream,
        &input,
        &weight,
        &scales,
        Some(&bias),
        &mut output,
    )?;
    assert_eq!(
        read(&context, &stream, &output)?,
        [33.0_f32, 94.0, 65.0, 190.0].map(bf16::from_f32)
    );
    Ok(())
}

#[test]
fn executes_gathered_mxfp4_matrix_bank() -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input = copy(&context, &stream, &pool, &[bf16::ONE; 64])?;
    let weight = [0x11_u8; 16]
        .into_iter()
        .chain([0x22_u8; 16])
        .chain([0x33_u8; 16])
        .chain([0x44_u8; 16])
        .collect::<Vec<_>>();
    let weight = copy(&context, &stream, &pool, &weight)?;
    let scales = copy(&context, &stream, &pool, &[127_u8; 4])?;
    let bias = [1.0_f32, 2.0, 3.0, 4.0].map(bf16::from_f32);
    let bias = copy(&context, &stream, &pool, &bias)?;
    let selected = copy(&context, &stream, &pool, &[1_u32, 0])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 4)?;
    MxFp4GatheredLinear::compile(&compiler, MxFp4GatheredSpec::new(2, 2, 32, 2)?)?.execute(
        &stream,
        &mut MxFp4GatheredOperands {
            input: &input,
            weight: &weight,
            scales: &scales,
            bias: Some(&bias),
            selected: &selected,
            output: &mut output,
        },
    )?;
    assert_eq!(
        read(&context, &stream, &output)?,
        [51.0_f32, 68.0, 17.0, 34.0].map(bf16::from_f32)
    );
    Ok(())
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

fn copy<T: DeviceElement + Copy>(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement + Copy>(
    context: &Context,
    stream: &Stream,
    values: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = context.allocate_pinned(values.len())?;
    stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
