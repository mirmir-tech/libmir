use std::path::PathBuf;

use libmir_cuda::{
    Result,
    kernels::{GptqLaunch, GptqLinear, GptqSpec},
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16, f16};

const INPUT: usize = 16;
const OUTPUT: usize = 16;
const GROUP: usize = 8;

#[test]
fn decodes_gptq_v1_and_v2_for_gemv_and_qmm() -> Result<()> {
    for legacy in [true, false] {
        for activation_order in [false, true] {
            for tokens in [1, 2] {
                check(tokens, legacy, activation_order)?;
            }
        }
    }
    Ok(())
}

fn check(tokens: usize, legacy: bool, activation_order: bool) -> Result<()> {
    let (context, stream, pool, compiler) = resources()?;
    let input_values = (0..tokens)
        .flat_map(|token| {
            (0..INPUT).map(move |feature| bf16::from_f32(input_value(token, feature)))
        })
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &packed_weights())?;
    let zero_points = copy_device(&context, &stream, &pool, &packed_zero_points(legacy))?;
    let scales = copy_device(&context, &stream, &pool, &scales())?;
    let group_indices = copy_device(
        &context,
        &stream,
        &pool,
        &(0..INPUT)
            .map(|feature| i32::try_from(group(feature, activation_order)).unwrap_or_default())
            .collect::<Vec<_>>(),
    )?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, tokens * OUTPUT)?;
    let operation =
        GptqLinear::compile(&compiler, GptqSpec::new(tokens, INPUT, OUTPUT, GROUP, legacy)?)?;
    operation.execute(
        &stream,
        &mut GptqLaunch {
            input: &input,
            weight: &weight,
            zero_points: &zero_points,
            scales: &scales,
            group_indices: &group_indices,
            output: &mut output,
        },
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = (0..tokens)
        .flat_map(|token| {
            (0..OUTPUT).map(move |row| {
                let sum = (0..INPUT)
                    .map(|feature| {
                        let group = group(feature, activation_order);
                        input_value(token, feature)
                            * (f32::from(quantized(row, feature)) - f32::from(zero(row, group)))
                            * scale(row, group)
                    })
                    .sum::<f32>();
                bf16::from_f32(sum)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

fn packed_weights() -> Vec<i32> {
    (0..INPUT / 8)
        .flat_map(|word| {
            (0..OUTPUT)
                .map(move |row| pack(std::array::from_fn(|lane| quantized(row, word * 8 + lane))))
        })
        .collect()
}

fn packed_zero_points(legacy: bool) -> Vec<i32> {
    (0..INPUT / GROUP)
        .flat_map(|group| {
            (0..OUTPUT / 8).map(move |word| {
                pack(std::array::from_fn(|lane| {
                    let zero = zero(word * 8 + lane, group);
                    if legacy {
                        zero.wrapping_sub(1) & 15
                    } else {
                        zero
                    }
                }))
            })
        })
        .collect()
}

fn scales() -> Vec<f16> {
    (0..INPUT / GROUP)
        .flat_map(|group| (0..OUTPUT).map(move |row| f16::from_f32(scale(row, group))))
        .collect()
}

fn pack(values: [u8; 8]) -> i32 {
    let packed = values
        .into_iter()
        .enumerate()
        .fold(0_u32, |word, (lane, value)| word | (u32::from(value) << (lane * 4)));
    i32::from_ne_bytes(packed.to_ne_bytes())
}

fn quantized(row: usize, feature: usize) -> u8 {
    u8::try_from((row * 3 + feature * 5) % 16).unwrap_or_default()
}

fn group(feature: usize, activation_order: bool) -> usize {
    if activation_order {
        (feature / 2) % (INPUT / GROUP)
    } else {
        feature / GROUP
    }
}

fn zero(row: usize, group: usize) -> u8 {
    u8::try_from(3 + (row + group) % 5).unwrap_or_default()
}

fn scale(row: usize, group: usize) -> f32 {
    if (row + group).is_multiple_of(2) {
        0.5
    } else {
        0.25
    }
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
