#![cfg(target_os = "linux")]

use std::path::PathBuf;

use libmir_cuda::{
    Result,
    kernels::{
        DirectFp8Activation, DirectFp8Format, DirectFp8Linear, DirectFp8Scale, DirectFp8Scales,
        DirectFp8Spec,
    },
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};

const TOKENS: usize = 2;
const INPUT: usize = 8;
const OUTPUT: usize = 2;

#[path = "direct_fp8/checkpoint.rs"]
mod checkpoint;
#[path = "direct_fp8/embedding.rs"]
mod embedding;
#[path = "direct_fp8/padded_grid.rs"]
mod padded_grid;
#[path = "direct_fp8/static_scale.rs"]
mod static_scale;

#[test]
fn executes_direct_e4m3_for_all_admitted_scale_contracts() -> Result<()> {
    check(DirectFp8Scale::Tensor, false, &[0.5])?;
    check(DirectFp8Scale::Tensor, true, &[2.0])?;
    check(DirectFp8Scale::OutputChannel, false, &[0.5, 0.25])?;
    check(
        DirectFp8Scale::BlockGrid {
            output_groups: 2,
            input_groups: 2,
            output_block_size: 1,
            input_block_size: 4,
        },
        false,
        &[0.5, 1.0, 0.25, 0.5],
    )?;
    check(
        DirectFp8Scale::BlockGrid {
            output_groups: 1,
            input_groups: 2,
            output_block_size: 2,
            input_block_size: 4,
        },
        false,
        &[0.5, 1.0],
    )?;
    check_bf16_scales_and_bias()
}

#[test]
fn executes_direct_e5m2_without_reinterpreting_it_as_e4m3() -> Result<()> {
    check_format(DirectFp8Format::E5M2, &weight_bytes_e5m2(), DirectFp8Scale::Tensor, &[0.5])
}

fn check(scale: DirectFp8Scale, inverse: bool, scales: &[f32]) -> Result<()> {
    check_format_with_inverse(DirectFp8Format::E4M3, &weight_bytes(), scale, inverse, scales)
}

fn check_format(
    format: DirectFp8Format,
    weights: &[u8],
    scale: DirectFp8Scale,
    scales: &[f32],
) -> Result<()> {
    check_format_with_inverse(format, weights, scale, false, scales)
}

fn check_format_with_inverse(
    format: DirectFp8Format,
    weights: &[u8],
    scale: DirectFp8Scale,
    inverse: bool,
    scales: &[f32],
) -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input_values = input_values().into_iter().map(bf16::from_f32).collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, weights)?;
    let device_scales = copy_device(&context, &stream, &pool, scales)?;
    let input_scale = copy_device(&context, &stream, &pool, &[1.0_f32])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, TOKENS * OUTPUT)?;
    let spec = DirectFp8Spec::new_with_format(
        format,
        TOKENS,
        INPUT,
        OUTPUT,
        scale,
        inverse,
        DirectFp8Activation::Bf16,
    )?;
    DirectFp8Linear::compile(&compiler, spec)?.execute(
        &stream,
        &input,
        &weight,
        DirectFp8Scales {
            weight: &device_scales,
            activation: &input_scale,
        },
        None,
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    assert_eq!(actual, expected(scale, inverse, scales));
    Ok(())
}

fn check_bf16_scales_and_bias() -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input_values = input_values().into_iter().map(bf16::from_f32).collect::<Vec<_>>();
    let scales = [bf16::from_f32(0.5), bf16::from_f32(0.25)];
    let biases = [bf16::from_f32(1.0), bf16::from_f32(-2.0)];
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &weight_bytes())?;
    let scales = copy_device(&context, &stream, &pool, &scales)?;
    let bias = copy_device(&context, &stream, &pool, &biases)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, TOKENS * OUTPUT)?;
    let spec = DirectFp8Spec::new(
        TOKENS,
        INPUT,
        OUTPUT,
        DirectFp8Scale::OutputChannel,
        false,
        DirectFp8Activation::Bf16,
    )?;
    DirectFp8Linear::compile(&compiler, spec)?.execute_bf16_scales(
        &stream,
        &input,
        &weight,
        DirectFp8Scales { weight: &scales, activation: &scales },
        Some(&bias),
        &mut output,
    )?;
    let actual = read(&context, &stream, &output)?;
    let mut expected = expected(DirectFp8Scale::OutputChannel, false, &[0.5, 0.25]);
    for values in expected.chunks_mut(OUTPUT) {
        values[0] = bf16::from_f32(values[0].to_f32() + 1.0);
        values[1] = bf16::from_f32(values[1].to_f32() - 2.0);
    }
    assert_eq!(actual, expected);
    Ok(())
}

fn expected(scale: DirectFp8Scale, inverse: bool, scales: &[f32]) -> Vec<bf16> {
    let input = input_values();
    let weight = weight_values();
    let mut expected = Vec::with_capacity(TOKENS * OUTPUT);
    for token in 0..TOKENS {
        for row in 0..OUTPUT {
            let mut total = 0.0_f32;
            for feature in 0..INPUT {
                let scale = match scale {
                    DirectFp8Scale::Tensor => scales[0],
                    DirectFp8Scale::OutputChannel => scales[row],
                    DirectFp8Scale::BlockGrid {
                        input_groups,
                        output_block_size,
                        input_block_size,
                        ..
                    } => {
                        let output_group = row / output_block_size;
                        let input_group = feature / input_block_size;
                        scales[output_group * input_groups + input_group]
                    },
                };
                let product = input[token * INPUT + feature] * weight[row * INPUT + feature];
                total += if inverse {
                    product / scale
                } else {
                    product * scale
                };
            }
            expected.push(bf16::from_f32(total));
        }
    }
    expected
}

fn input_values() -> Vec<f32> {
    vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
}

fn weight_values() -> Vec<f32> {
    vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, -1.0, 0.5, 0.0, 1.0, -2.0, 0.5, -0.5]
}

fn weight_bytes() -> Vec<u8> {
    vec![
        0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x40, 0xb8, 0x30, 0x00, 0x38, 0xc0, 0x30,
        0xb0,
    ]
}

fn weight_bytes_e5m2() -> Vec<u8> {
    vec![
        0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x40, 0xbc, 0x38, 0x00, 0x3c, 0xc0, 0x38,
        0xb8,
    ]
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
