use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use libmir_cuda::{
    Result,
    kernels::{PackedInt8Launch, PackedInt8Linear, PackedInt8Spec},
};
use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

#[test]
fn real_int4_checkpoint_projection_matches_direct_unpacking()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_INT4_MODEL") else {
        return Ok(());
    };
    let catalog = TensorCatalog::from_layout(&ModelLayout::inspect(Path::new(&root))?)?;
    let prefix = "model.layers.0.self_attn.q_proj";
    let weight_info = catalog.get(&format!("{prefix}.weight_packed")).ok_or("missing weight")?;
    let scale_info = catalog.get(&format!("{prefix}.weight_scale")).ok_or("missing scale")?;
    let weight = words(weight_info)?;
    let scales = bf16_values(scale_info)?;
    let (context, stream, pool, compiler) = resources()?;
    let input = copy_device(&context, &stream, &pool, &[bf16::from_f32(1.0); 1_024])?;
    let weight_device = copy_device(&context, &stream, &pool, &weight)?;
    let scale_device = copy_device(&context, &stream, &pool, &scales)?;
    let mut output = pool.allocate_zeroed::<bf16>(&stream, 2_048)?;
    let operation =
        PackedInt8Linear::compile(&compiler, PackedInt8Spec::new_packed(1, 1_024, 2_048, 4, 128)?)?;
    operation.execute(
        &stream,
        &mut PackedInt8Launch {
            input: &input,
            weight: &weight_device,
            scales: &scale_device,
            output: &mut output,
        },
    )?;
    let actual = read(&context, &stream, &output)?;
    for row in 0..2_048 {
        let sum = (0..1_024)
            .map(|feature| {
                let word = u32::from_ne_bytes(weight[row * 128 + feature / 8].to_ne_bytes());
                let packed = (word >> ((feature % 8) * 4)) & 0xf;
                (f32::from(u8::try_from(packed).unwrap_or_default()) - 8.0)
                    * scales[row * 8 + feature / 128].to_f32()
            })
            .sum::<f32>();
        assert_eq!(actual[row], bf16::from_f32(sum));
    }
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

fn words(info: &TensorInfo) -> std::io::Result<Vec<i32>> {
    Ok(payload(info)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect())
}

fn bf16_values(info: &TensorInfo) -> std::io::Result<Vec<bf16>> {
    Ok(payload(info)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| bf16::from_bits(u16::from_le_bytes(*word)))
        .collect())
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
