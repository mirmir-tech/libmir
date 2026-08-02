use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use libmir_cuda::{
    Result,
    kernels::{
        PackedInt8Embedding, PackedInt8EmbeddingLaunch, PackedInt8EmbeddingSpec, PackedInt8Launch,
        PackedInt8Linear, PackedInt8Spec,
    },
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

#[test]
fn decodes_compressed_tensors_int8_for_gemv_and_qmm() -> Result<()> {
    for tokens in [1, 2] {
        check(tokens)?;
    }
    Ok(())
}

#[test]
fn gathers_compressed_tensors_int8_embeddings() -> Result<()> {
    let driver = mircuda::Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let selected = copy_device(&context, &stream, &pool, &[0_u32, 1, 0])?;
    let weight = copy_device(&context, &stream, &pool, &embedding_weights())?;
    let scales =
        copy_device(&context, &stream, &pool, &[bf16::from_f32(0.5), bf16::from_f32(0.25)])?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 32)?;
    let compiler =
        Compiler::with_include_paths(context.clone(), [PathBuf::from("/usr/local/cuda/include")])?;
    let operation =
        PackedInt8Embedding::compile(&compiler, PackedInt8EmbeddingSpec::new(2, 16, 2.0)?)?;

    operation.execute(
        &stream,
        &mut PackedInt8EmbeddingLaunch {
            selected: &selected,
            selected_start: 1,
            tokens: 2,
            weight: &weight,
            scales: &scales,
            output: &mut output,
        },
    )?;

    let mut host = context.allocate_pinned::<bf16>(32)?;
    stream.copy_to_host(&output, &mut host)?;
    let row_one = [bf16::from_f32(1.0); 16];
    let row_zero = [-2.0_f32, -1.0, 0.0, 1.0].into_iter().cycle().take(16).map(bf16::from_f32);
    let expected = row_one.into_iter().chain(row_zero).collect::<Vec<_>>();
    assert_eq!(host.to_vec()?, expected);
    Ok(())
}

#[test]
fn real_checkpoint_projection_matches_direct_unpacking()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_INT8_MODEL") else {
        return Ok(());
    };
    let catalog = TensorCatalog::from_layout(&ModelLayout::inspect(Path::new(&root))?)?;
    let prefix = "model.layers.0.self_attn.q_proj";
    let weight_info = catalog.get(&format!("{prefix}.weight_packed")).ok_or("missing weight")?;
    let scale_info = catalog.get(&format!("{prefix}.weight_scale")).ok_or("missing scale")?;
    let weight_bytes = payload(weight_info)?;
    let weight = weight_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    let scale_bytes = payload(scale_info)?;
    let scales = scale_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)))
        .collect::<Vec<_>>();
    let input = vec![bf16::from_f32(1.0); 1_024];
    let (context, stream, pool, compiler) = resources()?;
    let input_device = copy_device(&context, &stream, &pool, &input)?;
    let weight_device = copy_device(&context, &stream, &pool, &weight)?;
    let scale_device = copy_device(&context, &stream, &pool, &scales)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 2_048)?;
    let operation = PackedInt8Linear::compile(&compiler, PackedInt8Spec::new(1, 1_024, 2_048)?)?;
    operation.execute(
        &stream,
        &mut PackedInt8Launch {
            input: &input_device,
            weight: &weight_device,
            scales: &scale_device,
            output: &mut output,
        },
    )?;
    let mut actual = context.allocate_pinned::<bf16>(2_048)?;
    stream.copy_to_host(&output, &mut actual)?;
    let actual = actual.to_vec()?;
    for row in 0..2_048 {
        let sum = (0..1_024)
            .map(|feature| {
                let word = weight[row * 256 + feature / 4].to_le_bytes()[feature % 4];
                f32::from(word) - 128.0
            })
            .sum::<f32>();
        assert_eq!(actual[row], bf16::from_f32(sum * scales[row].to_f32()));
    }
    Ok(())
}

fn check(tokens: usize) -> Result<()> {
    let driver = mircuda::Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let input_values = (0..tokens)
        .flat_map(|token| (0..16).map(move |feature| bf16::from_f32(input_value(token, feature))))
        .collect::<Vec<_>>();
    let input = copy_device(&context, &stream, &pool, &input_values)?;
    let weight = copy_device(&context, &stream, &pool, &projection_weights())?;
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
    let expected = (0..tokens)
        .flat_map(|token| {
            [0, 1].map(|row| {
                let scale = [0.5_f32, 0.25][row];
                let value = (0..16)
                    .map(|feature| input_value(token, feature) * f32::from(quantized(row, feature)))
                    .sum::<f32>();
                bf16::from_f32(value * scale)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(host.to_vec()?, expected);
    Ok(())
}

fn embedding_weights() -> Vec<i32> {
    let first = pack([-2, -1, 0, 1]);
    let second = pack([2, 2, 2, 2]);
    vec![first; 4].into_iter().chain(vec![second; 4]).collect()
}

fn projection_weights() -> Vec<i32> {
    (0..2)
        .flat_map(|row| {
            (0..4)
                .map(move |word| pack(std::array::from_fn(|byte| quantized(row, word * 4 + byte))))
        })
        .collect()
}

fn quantized(row: usize, feature: usize) -> i8 {
    if row == 0 {
        i8::try_from(feature).unwrap_or_default() - 8
    } else if feature.is_multiple_of(2) {
        3
    } else {
        -2
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

fn payload(info: &TensorInfo) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(
        info.data_start
            .checked_add(info.data_offsets[0])
            .ok_or_else(|| std::io::Error::other("offset"))?,
    ))?;
    let mut bytes = vec![
        0;
        usize::try_from(info.data_offsets[1] - info.data_offsets[0])
            .map_err(std::io::Error::other)?
    ];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
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
