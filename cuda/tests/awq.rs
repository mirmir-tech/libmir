use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use libmir_cuda::{
    Result,
    kernels::{AwqLaunch, AwqLinear, AwqSpec},
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16, f16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

const INPUT: usize = 16;
const OUTPUT: usize = 16;
const GROUP: usize = 8;

#[test]
fn decodes_awq_gemm_for_gemv_and_qmm() -> Result<()> {
    for tokens in [1, 2] {
        check(tokens)?;
    }
    Ok(())
}

#[test]
fn real_checkpoint_projection_matches_awq_unpacking()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_AWQ_MODEL") else {
        return Ok(());
    };
    let catalog = TensorCatalog::from_layout(&ModelLayout::inspect(Path::new(&root))?)?;
    let prefix = "model.layers.0.self_attn.q_proj";
    let info = |suffix| catalog.get(&format!("{prefix}.{suffix}")).ok_or("missing AWQ tensor");
    let weight = i32_payload(info("qweight")?)?;
    let zeros = i32_payload(info("qzeros")?)?;
    let scales = f16_payload(info("scales")?)?;
    let (context, stream, pool, compiler) = resources()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::ONE; 1_024])?;
    let weight_device = copy_device(&context, &stream, &pool, &weight)?;
    let zero_device = copy_device(&context, &stream, &pool, &zeros)?;
    let scale_device = copy_device(&context, &stream, &pool, &scales)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 2_048)?;
    AwqLinear::compile(&compiler, AwqSpec::new(1, 1_024, 2_048, 128)?)?.execute(
        &stream,
        &mut AwqLaunch {
            input: &input,
            weight: &weight_device,
            zero_points: &zero_device,
            scales: &scale_device,
            output: &mut output,
        },
    )?;
    let actual = read(&context, &stream, &output)?;
    for row in 0..2_048 {
        let expected = (0..1_024)
            .map(|feature| {
                let shift = shift(row % 8);
                let value = (u32::from_ne_bytes(weight[feature * 256 + row / 8].to_ne_bytes())
                    >> shift)
                    & 15;
                let zero = (u32::from_ne_bytes(zeros[feature / 128 * 256 + row / 8].to_ne_bytes())
                    >> shift)
                    & 15;
                (f32::from(value.to_le_bytes()[0]) - f32::from(zero.to_le_bytes()[0]))
                    * scales[feature / 128 * 2_048 + row].to_f32()
            })
            .sum::<f32>();
        assert_eq!(actual[row], bf16::from_f32(expected), "row {row}");
    }
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
    let weight = copy_device(&context, &stream, &pool, &packed_weights())?;
    let zero_points = copy_device(&context, &stream, &pool, &packed_zero_points())?;
    let scales = copy_device(&context, &stream, &pool, &scales())?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, tokens * OUTPUT)?;
    let operation = AwqLinear::compile(&compiler, AwqSpec::new(tokens, INPUT, OUTPUT, GROUP)?)?;
    operation.execute(
        &stream,
        &mut AwqLaunch {
            input: &input,
            weight: &weight,
            zero_points: &zero_points,
            scales: &scales,
            output: &mut output,
        },
    )?;
    let actual = read(&context, &stream, &output)?;
    let expected = (0..tokens)
        .flat_map(|token| {
            (0..OUTPUT).map(move |row| {
                let sum = (0..INPUT)
                    .map(|feature| {
                        input_value(token, feature)
                            * (f32::from(quantized(row, feature)) - f32::from(zero(row, feature)))
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

fn packed_weights() -> Vec<i32> {
    (0..INPUT)
        .flat_map(|feature| {
            (0..OUTPUT / 8).map(move |word| {
                pack(std::array::from_fn(|lane| quantized(word * 8 + lane, feature)))
            })
        })
        .collect()
}

fn packed_zero_points() -> Vec<i32> {
    (0..INPUT / GROUP)
        .flat_map(|group| {
            (0..OUTPUT / 8).map(move |word| {
                pack(std::array::from_fn(|lane| zero(word * 8 + lane, group * GROUP)))
            })
        })
        .collect()
}

fn scales() -> Vec<f16> {
    (0..INPUT / GROUP)
        .flat_map(|group| (0..OUTPUT).map(move |row| f16::from_f32(scale(row, group * GROUP))))
        .collect()
}

fn pack(values: [u8; 8]) -> i32 {
    let packed = values
        .into_iter()
        .enumerate()
        .fold(0_u32, |word, (row, value)| word | (u32::from(value) << shift(row)));
    i32::from_ne_bytes(packed.to_ne_bytes())
}

const fn shift(row: usize) -> usize {
    (((row & 1) << 2) | (row >> 1)) * 4
}

fn quantized(row: usize, feature: usize) -> u8 {
    u8::try_from((row * 3 + feature * 5) % 16).unwrap_or_default()
}

fn zero(row: usize, feature: usize) -> u8 {
    u8::try_from(3 + (row + feature / GROUP) % 5).unwrap_or_default()
}

fn scale(row: usize, feature: usize) -> f32 {
    if (row + feature / GROUP).is_multiple_of(2) {
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

fn i32_payload(info: &TensorInfo) -> std::io::Result<Vec<i32>> {
    Ok(payload(info)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect())
}

fn f16_payload(info: &TensorInfo) -> std::io::Result<Vec<f16>> {
    Ok(payload(info)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)))
        .collect())
}

fn payload(info: &TensorInfo) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(info.data_start + info.data_offsets[0]))?;
    let bytes_len = usize::try_from(info.data_offsets[1] - info.data_offsets[0])
        .map_err(std::io::Error::other)?;
    let mut bytes = vec![0; bytes_len];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
