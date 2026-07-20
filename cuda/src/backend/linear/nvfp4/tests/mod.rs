mod pack;

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

use super::{native::NativeNvFp4, reference::ReferenceNvFp4, *};
use crate::{CudaConfig, CudaTensorSet};

const BASE: &str = "model.language_model.layers.0.experts.0.gate_proj";

#[test]
fn checkpoint_native_projection_matches_reference() -> Result<()> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let weight_info = required(&catalog, &format!("{BASE}.weight"))?;
    let scale_info = required(&catalog, &format!("{BASE}.weight_scale"))?;
    let global_info = required(&catalog, &format!("{BASE}.weight_scale_2"))?;
    let input_scale_info = required(&catalog, &format!("{BASE}.input_scale"))?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, [weight_info, scale_info, global_info, input_scale_info])?;
    let weight = tensors
        .get(&weight_info.name)
        .ok_or_else(|| Error::MissingTensor(weight_info.name.clone()))?;
    let scale = tensors
        .get(&scale_info.name)
        .ok_or_else(|| Error::MissingTensor(scale_info.name.clone()))?;
    let global = tensors
        .get(&global_info.name)
        .ok_or_else(|| Error::MissingTensor(global_info.name.clone()))?;
    let input_scale = tensors
        .get(&input_scale_info.name)
        .ok_or_else(|| Error::MissingTensor(input_scale_info.name.clone()))?;
    let config = NvFp4Config::new(weight_info.shape[1] * 2, weight_info.shape[0]);
    let checkpoint = NvFp4Tensors {
        weight,
        weight_scale: scale,
        weight_scale_2: global,
        input_scale,
    };
    let mut reference = ReferenceNvFp4::new(&backend, 1, config, checkpoint)?;
    let mut native = NativeNvFp4::new(&backend, 1, config, checkpoint)?;
    let packed = read_payload(weight_info)?;
    let scales = read_payload(scale_info)?;
    let global_scale = read_scalar_f32(global_info)?;
    check_materialized(&backend, &reference, &packed, &scales, global_scale)?;
    check_projection(&backend, &mut reference, &mut native, config, &packed, &scales, global_scale)
}

fn upload(backend: &CudaBackend, infos: [&TensorInfo; 4]) -> Result<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    for info in infos {
        upload.enqueue(info)?;
    }
    upload.finish()
}

fn check_materialized(
    backend: &CudaBackend,
    linear: &ReferenceNvFp4,
    packed: &[u8],
    scales: &[u8],
    global: f32,
) -> Result<()> {
    let actual = read_device(backend, &linear.weight)?;
    for index in (0..actual.len()).step_by(997) {
        let expected = bf16::from_f32(dequant(index, packed, scales, global));
        assert_eq!(actual[index], expected, "dequantized element {index}");
    }
    Ok(())
}

fn check_projection(
    backend: &CudaBackend,
    reference: &mut ReferenceNvFp4,
    native: &mut NativeNvFp4,
    config: NvFp4Config,
    packed: &[u8],
    scales: &[u8],
    global: f32,
) -> Result<()> {
    let input_values = (0..config.input_features)
        .map(|index| {
            let value = f32::from(u8::try_from(index % 31)?) / 16.0 - 0.9375;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    let input = copy_device(backend, &input_values)?;
    let mut reference_output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, reference.output_elements()?)?;
    let mut native_output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, native.output_elements()?)?;
    reference.execute(&input, &mut reference_output)?;
    native.execute(&input, &mut native_output)?;
    let actual = read_device(backend, &reference_output)?;
    let native_actual = read_device(backend, &native_output)?;
    for (row, value) in actual.iter().enumerate().take(16) {
        let offset = row * config.input_features;
        let expected = input_values.iter().enumerate().fold(0.0_f32, |sum, (column, input)| {
            let weight = bf16::from_f32(dequant(offset + column, packed, scales, global));
            input.to_f32().mul_add(weight.to_f32(), sum)
        });
        let tolerance = (expected.abs() * 0.02).max(0.125);
        assert!((value.to_f32() - expected).abs() <= tolerance);
    }
    check_quality(&actual, &native_actual);
    Ok(())
}

fn check_quality(reference: &[bf16], native: &[bf16]) {
    let non_finite = native
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (!value.to_f32().is_finite()).then_some(index))
        .collect::<Vec<_>>();
    assert!(non_finite.is_empty(), "NVFP4 output contains non-finite rows {non_finite:?}");
    let (error, energy, dot, native_energy) = reference.iter().zip(native).fold(
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
        |(error, energy, dot, native_energy), (reference, native)| {
            let reference = f64::from(reference.to_f32());
            let native = f64::from(native.to_f32());
            (
                (native - reference).mul_add(native - reference, error),
                reference.mul_add(reference, energy),
                reference.mul_add(native, dot),
                native.mul_add(native, native_energy),
            )
        },
    );
    let normalized_error = (error / energy).sqrt();
    let cosine = dot / (energy * native_energy).sqrt();
    assert!(normalized_error < 0.20, "NVFP4 normalized error {normalized_error}");
    assert!(cosine > 0.97, "NVFP4 cosine {cosine}");
}

fn dequant(index: usize, packed: &[u8], scales: &[u8], global: f32) -> f32 {
    let byte = packed[index / 2];
    let nibble = if index.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    };
    fp4(nibble) * fp8_e4m3(scales[index / 16]) * global
}

fn fp4(value: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = VALUES[usize::from(value & 0x07)];
    if value & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn fp8_e4m3(value: u8) -> f32 {
    let sign = if value & 0x80 == 0 {
        1.0
    } else {
        -1.0
    };
    let exponent = i32::from((value >> 3) & 0x0f);
    let mantissa = value & 0x07;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(exponent - 7)
    }
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}

fn read_payload(info: &TensorInfo) -> Result<Vec<u8>> {
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(info.payload_start()?))?;
    let mut bytes = vec![0; info.payload_bytes()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_scalar_f32(info: &TensorInfo) -> Result<f32> {
    let payload = read_payload(info)?;
    if payload.len() != 4 {
        return Err(Error::InvalidTensorSize {
            name: info.name.clone(),
            expected: 4,
            actual: payload.len(),
        });
    }
    let mut bytes = [0; 4];
    bytes.copy_from_slice(&payload);
    Ok(f32::from_le_bytes(bytes))
}

fn copy_device<T: DeviceElement + Copy>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.inner.stream.synchronize()?;
    Ok(device)
}

fn read_device<T: DeviceElement + Copy>(
    backend: &CudaBackend,
    device: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(device.len())?;
    backend.inner.stream.copy_to_host(device, &mut host)?;
    Ok(host.to_vec()?)
}
