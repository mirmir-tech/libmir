#[cfg(target_os = "linux")]
use std::fs;

#[cfg(target_os = "linux")]
use mircuda::{bf16, f16};
#[cfg(target_os = "linux")]
use models::weights::TensorInfo;

#[cfg(target_os = "linux")]
use super::CudaTensorDType;
#[cfg(target_os = "linux")]
use crate::{CudaBackend, CudaConfig, Result};

#[test]
#[cfg(target_os = "linux")]
fn persistently_casts_dense_f16_and_f32_to_bf16_on_device() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let cast = backend.prepare_dense_cast()?;
    for dtype in ["F16", "F32"] {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-dense-cast-{}-{dtype}.bin", std::process::id()));
        let bytes = dense_bytes(dtype);
        fs::write(&path, &bytes)?;
        let info = TensorInfo {
            name: format!("dense.{dtype}"),
            file: path.clone(),
            dtype: dtype.into(),
            shape: vec![3],
            data_start: 0,
            data_offsets: [0, u64::try_from(bytes.len())?],
        };
        let mut upload = backend.begin_tensor_upload();
        upload.enqueue_as_bf16(&info, &cast)?;
        let tensors = upload.finish()?;
        let tensor = tensors
            .get(&info.name)
            .ok_or_else(|| crate::Error::MissingTensor(info.name.clone()))?;
        assert_eq!(tensor.dtype(), CudaTensorDType::Bf16);
        assert_eq!(tensors.read_bf16(&info.name)?, [1.0_f32, -2.0, 3.5].map(bf16::from_f32));
        fs::remove_file(path)?;
    }
    Ok(())
}

#[test]
#[cfg(target_os = "linux")]
fn flushes_bounded_staging_without_losing_device_tensors() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut paths = Vec::new();
    let mut infos = Vec::new();
    for (index, value) in [1.0_f32, -2.0].into_iter().enumerate() {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-bounded-upload-{}-{index}.bin", std::process::id()));
        fs::write(&path, bf16::from_f32(value).to_bits().to_le_bytes())?;
        infos.push(TensorInfo {
            name: format!("bounded.{index}"),
            file: path.clone(),
            dtype: "BF16".into(),
            shape: vec![1],
            data_start: 0,
            data_offsets: [0, 2],
        });
        paths.push(path);
    }
    let mut upload = backend.begin_tensor_upload();
    upload.set_staging_limit(1);
    for info in &infos {
        upload.enqueue(info)?;
    }
    let tensors = upload.finish()?;
    assert_eq!(tensors.read_bf16("bounded.0")?, [bf16::from_f32(1.0)]);
    assert_eq!(tensors.read_bf16("bounded.1")?, [bf16::from_f32(-2.0)]);
    for path in paths {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn dense_bytes(dtype: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [1.0_f32, -2.0, 3.5] {
        match dtype {
            "F16" => bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes()),
            "F32" => bytes.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!("test dtype"),
        }
    }
    bytes
}
