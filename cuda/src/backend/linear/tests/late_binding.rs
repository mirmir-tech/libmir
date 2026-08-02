use std::fs;

use mircuda::bf16;
use models::weights::{
    CompressedIntegerActivationOrder, CompressedIntegerBits, CompressedIntegerPacking,
    CompressedIntegerQuantization, CompressedIntegerScaleDType, CompressedIntegerScaleStrategy,
    CompressedIntegerSignedness, CompressedIntegerStorageDType, CompressedIntegerZeroPointMode,
    LogicalTensorRole, TensorBinding, TensorInfo, TensorStorage,
};

use super::{append_bf16, append_packed_row};
use crate::{
    AffineQuantizedWeight, CompressedInt8Bf16Linear, CompressedInt8Weight, CudaBackend, CudaConfig,
    Result, backend::linear::AffineProjection,
};

const INPUT: usize = 64;
const OUTPUT: usize = 2;
const TOKENS: usize = 2;

#[test]
fn shared_affine_plan_uses_the_current_layer_weight() -> Result<()> {
    let path = std::env::temp_dir()
        .join(format!("libmir-cuda-affine-late-binding-{}.bin", std::process::id()));
    let mut bytes = Vec::new();
    let mut infos = Vec::new();
    for (layer, values) in [[1, 2], [3, 4]].into_iter().enumerate() {
        let start = u64::try_from(bytes.len())?;
        for value in values {
            append_packed_row(&mut bytes, value, 4, INPUT);
        }
        let weight_end = u64::try_from(bytes.len())?;
        append_bf16(&mut bytes, [1.0, 1.0]);
        let scale_end = u64::try_from(bytes.len())?;
        append_bf16(&mut bytes, [0.0, 0.0]);
        let bias_end = u64::try_from(bytes.len())?;
        let prefix = format!("layer.{layer}");
        infos.extend([
            info(&format!("{prefix}.weight"), &path, "U32", start, weight_end),
            info(&format!("{prefix}.scales"), &path, "BF16", weight_end, scale_end),
            info(&format!("{prefix}.biases"), &path, "BF16", scale_end, bias_end),
        ]);
    }
    fs::write(&path, bytes)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for tensor in &infos {
        upload.enqueue(tensor)?;
    }
    let tensors = upload.finish()?;
    let weights = [
        AffineQuantizedWeight::load(&tensors, "layer.0")?,
        AffineQuantizedWeight::load(&tensors, "layer.1")?,
    ];
    let projection = AffineProjection::new(&backend, TOKENS, INPUT, OUTPUT, 64, 4, &weights[0])?;
    let input = device_input(&backend)?;
    let mut output = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, TOKENS * OUTPUT)?;
    projection.execute(&input, &weights[1], &mut output)?;
    let mut host = backend.inner.context.allocate_pinned::<bf16>(TOKENS * OUTPUT)?;
    backend.inner.stream.copy_to_host(&output, &mut host)?;
    assert_eq!(host.to_vec()?, [192.0, 256.0, 192.0, 256.0].map(bf16::from_f32));
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn shared_int8_plan_uses_the_current_layer_weight() -> Result<()> {
    let path = std::env::temp_dir()
        .join(format!("libmir-cuda-int8-late-binding-{}.bin", std::process::id()));
    let mut bytes = Vec::new();
    let mut infos = Vec::new();
    let mut bindings = Vec::new();
    for (layer, values) in [[1_i8, 2], [3, 4]].into_iter().enumerate() {
        let start = u64::try_from(bytes.len())?;
        for value in values {
            append_int8_row(&mut bytes, value);
        }
        let weight_end = u64::try_from(bytes.len())?;
        append_bf16(&mut bytes, [1.0, 1.0]);
        let scale_end = u64::try_from(bytes.len())?;
        let prefix = format!("layer.{layer}");
        infos.extend([
            info(&format!("{prefix}.weight"), &path, "I32", start, weight_end),
            info(&format!("{prefix}.scales"), &path, "BF16", weight_end, scale_end),
        ]);
        bindings.push(int8_binding(&prefix));
    }
    fs::write(&path, bytes)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for tensor in &infos {
        upload.enqueue(tensor)?;
    }
    let tensors = upload.finish()?;
    let weights = [
        CompressedInt8Weight::load_binding(&tensors, &bindings[0], INPUT, OUTPUT)?,
        CompressedInt8Weight::load_binding(&tensors, &bindings[1], INPUT, OUTPUT)?,
    ];
    let projection = CompressedInt8Bf16Linear::new(&backend, TOKENS, INPUT, OUTPUT, &weights[0])?;
    let input = device_input(&backend)?;
    let mut output = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, TOKENS * OUTPUT)?;
    projection.execute(&input, &weights[1], &mut output)?;
    let mut host = backend.inner.context.allocate_pinned::<bf16>(TOKENS * OUTPUT)?;
    backend.inner.stream.copy_to_host(&output, &mut host)?;
    assert_eq!(host.to_vec()?, [192.0, 256.0, 192.0, 256.0].map(bf16::from_f32));
    fs::remove_file(path)?;
    Ok(())
}

fn info(name: &str, path: &std::path::Path, dtype: &str, start: u64, end: u64) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.into(),
        dtype: dtype.into(),
        shape: vec![
            OUTPUT,
            match dtype {
                "U32" => INPUT * 4 / 32,
                "I32" => INPUT / 4,
                _ => 1,
            },
        ],
        data_start: 0,
        data_offsets: [start, end],
    }
}

fn int8_binding(prefix: &str) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Auxiliary { path: prefix.into() },
        source: format!("{prefix}.weight"),
        shape: vec![OUTPUT, INPUT / 4],
        logical_shape: Some(vec![OUTPUT, INPUT]),
        transforms: Vec::new(),
        storage: TensorStorage::PackedInt8 {
            format: CompressedIntegerQuantization {
                bits: CompressedIntegerBits::Eight,
                scale_strategy: CompressedIntegerScaleStrategy::Channel,
                signedness: CompressedIntegerSignedness::OffsetBinary,
                zero_point: CompressedIntegerZeroPointMode::None,
                activation_order: CompressedIntegerActivationOrder::None,
                packing: CompressedIntegerPacking::DenseLittleEndian,
                storage_dtype: CompressedIntegerStorageDType::I32,
                scale_dtype: CompressedIntegerScaleDType::BF16,
            },
            scales: format!("{prefix}.scales"),
            shape: format!("{prefix}.shape"),
            zero_points: None,
            group_indices: None,
        },
    }
}

fn append_int8_row(bytes: &mut Vec<u8>, value: i8) {
    let packed = i32::from_le_bytes([value.to_ne_bytes()[0].wrapping_add(128); 4]);
    for _ in 0..INPUT / 4 {
        bytes.extend_from_slice(&packed.to_le_bytes());
    }
}

fn device_input(backend: &CudaBackend) -> Result<mircuda::DeviceBuffer<bf16>> {
    let values = [bf16::ONE; TOKENS * INPUT];
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(&values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
