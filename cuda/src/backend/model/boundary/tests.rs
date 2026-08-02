use std::{
    fs,
    path::{Path, PathBuf},
};

use mircuda::bf16;
use models::weights::{
    BindingTransform, CompressedIntegerActivationOrder, CompressedIntegerBits,
    CompressedIntegerPacking, CompressedIntegerQuantization, CompressedIntegerScaleDType,
    CompressedIntegerScaleStrategy, CompressedIntegerSignedness, CompressedIntegerStorageDType,
    CompressedIntegerZeroPointMode, Float8Format, Float8Quantization, LogicalTensorRole,
    TensorBinding, TensorInfo, TensorStorage,
};
use runtime::backend::SamplingLogits;

use super::{ModelEmbeddingTemplate, ModelOutputHeadTemplate};
use crate::{
    CudaBackend, CudaConfig, CudaTensorSet, Result, backend::linear::CheckpointProjectionWeight,
};

#[path = "tests/mxfp4.rs"]
mod mxfp4;
#[path = "tests/mxfp4_moe.rs"]
mod mxfp4_moe;
#[path = "tests/mxfp8.rs"]
mod mxfp8;
#[path = "tests/mxfp8_moe.rs"]
mod mxfp8_moe;

#[test]
fn executes_packed_int8_model_boundaries_without_dense_weights() -> Result<()> {
    let path = fixture_path();
    let infos = write_fixture(&path)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let binding = binding();
    let weight = CheckpointProjectionWeight::load_binding(&tensors, &binding)?;

    let embedding =
        ModelEmbeddingTemplate::new(weight.clone(), 2, 16, 2.0)?.instantiate(&backend)?;
    let selected = copy(&backend, &[1_u32])?;
    let mut embedded = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 16)?;
    embedding.execute(&selected, 0, &mut embedded)?;
    assert_eq!(read(&backend, &embedded)?, [bf16::from_f32(1.0); 16]);

    let mut output =
        ModelOutputHeadTemplate::prepare(&backend, weight, 16, 2)?.instantiate(&backend)?;
    let input = copy(&backend, &[bf16::from_f32(1.0); 16])?;
    let mut logits = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 2)?;
    output.execute(&input, &mut logits, SamplingLogits::Full)?;
    assert_eq!(read(&backend, &logits)?, [bf16::from_f32(-4.0), bf16::from_f32(8.0)]);
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn executes_unscaled_e5m2_output_head_without_dense_weights() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-e5m2-boundary-{}.bin", std::process::id()));
    let bytes = [
        0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x40, 0xbc, 0x38, 0x00, 0x3c, 0xc0, 0x38,
        0xb8,
    ];
    fs::write(&path, bytes)?;
    let info = info("boundary.weight", &path, "F8_E5M2", vec![2, 8], 0, 16);
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &[info])?;
    let binding = TensorBinding {
        role: LogicalTensorRole::Output,
        source: "boundary.weight".into(),
        shape: vec![2, 8],
        logical_shape: Some(vec![2, 8]),
        transforms: Vec::new(),
        storage: TensorStorage::Float8 {
            format: Float8Quantization::unscaled(Float8Format::E5M2),
            scale: None,
            input_scale: None,
            bias: None,
        },
    };
    let weight = CheckpointProjectionWeight::load_binding(&tensors, &binding)?;

    let embedding =
        ModelEmbeddingTemplate::new(weight.clone(), 2, 8, 2.0)?.instantiate(&backend)?;
    let selected = copy(&backend, &[1_u32, 0])?;
    let mut embedded = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 16)?;
    embedding.execute_batch(&selected, 0, 2, &mut embedded)?;
    let expected =
        [4.0, -2.0, 1.0, 0.0, 2.0, -4.0, 1.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            .map(bf16::from_f32);
    assert_eq!(read(&backend, &embedded)?, expected);

    let mut output =
        ModelOutputHeadTemplate::prepare(&backend, weight, 8, 2)?.instantiate(&backend)?;
    let input = copy(&backend, &[bf16::from_f32(1.0); 8])?;
    let mut logits = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 2)?;
    output.execute(&input, &mut logits, SamplingLogits::Full)?;
    assert_eq!(read(&backend, &logits)?, [bf16::from_f32(8.0), bf16::from_f32(0.5)]);
    fs::remove_file(path)?;
    Ok(())
}

fn write_fixture(path: &PathBuf) -> Result<[TensorInfo; 2]> {
    let mut bytes = Vec::new();
    for word in weights() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let weight_end = u64::try_from(bytes.len())?;
    for scale in [0.5_f32, 0.25] {
        bytes.extend_from_slice(&bf16::from_f32(scale).to_bits().to_le_bytes());
    }
    let scale_end = u64::try_from(bytes.len())?;
    fs::write(path, bytes)?;
    Ok([
        info("boundary.weight_packed", path, "I32", vec![2, 4], 0, weight_end),
        info("boundary.weight_scale", path, "BF16", vec![2, 1], weight_end, scale_end),
    ])
}

fn binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Embedding,
        source: "boundary.weight_packed".into(),
        shape: vec![2, 4],
        logical_shape: Some(vec![2, 16]),
        transforms: Vec::<BindingTransform>::new(),
        storage: TensorStorage::PackedInt8 {
            format: format(),
            scales: "boundary.weight_scale".into(),
            shape: "boundary.weight_shape".into(),
            zero_points: None,
            group_indices: None,
        },
    }
}

fn format() -> CompressedIntegerQuantization {
    CompressedIntegerQuantization {
        bits: CompressedIntegerBits::Eight,
        scale_strategy: CompressedIntegerScaleStrategy::Channel,
        signedness: CompressedIntegerSignedness::OffsetBinary,
        zero_point: CompressedIntegerZeroPointMode::None,
        activation_order: CompressedIntegerActivationOrder::None,
        packing: CompressedIntegerPacking::DenseLittleEndian,
        storage_dtype: CompressedIntegerStorageDType::I32,
        scale_dtype: CompressedIntegerScaleDType::BF16,
    }
}

fn weights() -> Vec<i32> {
    let first = pack([-2, -1, 0, 1]);
    let second = pack([2, 2, 2, 2]);
    vec![first; 4].into_iter().chain(vec![second; 4]).collect()
}

fn pack(values: [i8; 4]) -> i32 {
    i32::from_le_bytes(values.map(|value| value.to_ne_bytes()[0].wrapping_add(128)))
}

fn info(
    name: &str,
    path: &Path,
    dtype: &str,
    shape: Vec<usize>,
    start: u64,
    end: u64,
) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.to_path_buf(),
        dtype: dtype.into(),
        shape,
        data_start: 0,
        data_offsets: [start, end],
    }
}

fn upload(backend: &CudaBackend, infos: &[TensorInfo]) -> Result<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    for info in infos {
        upload.enqueue(info)?;
    }
    upload.finish()
}

fn copy<T: mircuda::DeviceElement + Copy>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<mircuda::DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: mircuda::DeviceElement + Copy>(
    backend: &CudaBackend,
    values: &mircuda::DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    backend.inner.stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}

fn fixture_path() -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-int8-boundary-{}.bin", std::process::id()))
}
