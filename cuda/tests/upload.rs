#[cfg(target_os = "linux")]
use std::{fs, path::PathBuf};

#[cfg(target_os = "linux")]
use libmir_cuda::{CudaBackend, CudaConfig, CudaTensorDType};
#[cfg(target_os = "linux")]
use mircuda::bf16;
#[cfg(target_os = "linux")]
use models::weights::TensorInfo;

#[test]
#[cfg(target_os = "linux")]
fn uploads_a_safetensors_range_without_an_intermediate_value_vector() -> libmir_cuda::Result<()> {
    let values = [1.0_f32, -2.0, 3.5, 4.25].map(bf16::from_f32);
    let mut file_bytes = vec![0_u8; 16];
    for value in values {
        file_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let path = temp_path();
    fs::write(&path, file_bytes)?;
    let info = TensorInfo {
        name: "model.test.weight".into(),
        file: path.clone(),
        dtype: "BF16".into(),
        shape: vec![2, 2],
        data_start: 8,
        data_offsets: [8, 16],
    };

    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut batch = backend.begin_tensor_upload();
    batch.enqueue(&info)?;
    let tensors = batch.finish()?;
    let tensor = tensors
        .get(&info.name)
        .ok_or(libmir_cuda::Error::MissingTensor(info.name.clone()))?;
    assert_eq!(tensor.dtype(), CudaTensorDType::Bf16);
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensors.read_bf16(&info.name)?, values);
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn temp_path() -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-upload-{}.bin", std::process::id()))
}
