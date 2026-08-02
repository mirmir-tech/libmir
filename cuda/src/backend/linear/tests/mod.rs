use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::weights::TensorInfo;

use super::*;
use crate::{CudaConfig, CudaTensor, CudaTensorDType, CudaTensorSet};

#[cfg(target_os = "linux")]
mod bitsandbytes;
mod late_binding;

#[test]
#[cfg(target_os = "linux")]
fn matches_a_dense_checkpoint_projection() -> Result<()> {
    let weights =
        [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0].map(bf16::from_f32);
    let mut weight_bytes = Vec::with_capacity(24);
    for value in weights {
        weight_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let path = temp_path();
    fs::write(&path, weight_bytes)?;
    let weight_info = TensorInfo {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        file: path.clone(),
        dtype: "BF16".into(),
        shape: vec![4, 3],
        data_start: 0,
        data_offsets: [0, 24],
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    upload.enqueue(&weight_info)?;
    let tensors = upload.finish()?;
    let weight = tensors
        .get(&weight_info.name)
        .ok_or_else(|| Error::MissingTensor(weight_info.name.clone()))?;
    assert_eq!(weight.dtype(), CudaTensorDType::Bf16);

    let input_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].map(bf16::from_f32);
    let mut host_input = backend.inner.context.allocate_pinned::<bf16>(6)?;
    host_input.copy_from_slice(&input_values)?;
    let mut input = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, 6)?;
    backend.inner.stream.copy_to_device(&mut host_input, &mut input)?;
    let mut linear = backend.prepare_bf16_linear(2, 3, 4)?;
    let mut output = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, linear.output_elements()?)?;
    linear.execute(&input, weight, &mut output)?;
    let mut host_output = backend.inner.context.allocate_pinned::<bf16>(8)?;
    backend.inner.stream.copy_to_host(&output, &mut host_output)?;

    let expected = [1.0_f32, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0].map(bf16::from_f32);
    assert_eq!(host_output.to_vec()?, expected);
    fs::remove_file(path)?;
    Ok(())
}

#[test]
#[cfg(target_os = "linux")]
fn matches_native_mlx_affine_expert_projections() -> Result<()> {
    for bits in [2, 3, 4, 5, 6, 8] {
        check_affine_expert(bits)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn check_affine_expert(bits: usize) -> Result<()> {
    const INPUT: usize = 64;
    const OUTPUT: usize = 2;
    const MATRICES: usize = 2;
    let words_per_row = INPUT * bits / 32;
    let mut bytes = Vec::new();
    for matrix in 0..MATRICES {
        for row in 0..OUTPUT {
            let quantized = if matrix == 0 {
                0_u32
            } else if row == 0 {
                1
            } else {
                2
            };
            append_packed_row(&mut bytes, quantized, bits, INPUT);
        }
    }
    let weight_end = u64::try_from(bytes.len())?;
    append_bf16(&mut bytes, [1.0, 1.0, 0.5, 0.25]);
    let scale_end = u64::try_from(bytes.len())?;
    append_bf16(&mut bytes, [0.0, 0.0, -1.0, 0.5]);
    let bias_end = u64::try_from(bytes.len())?;
    let path = temp_path_for(bits);
    fs::write(&path, bytes)?;
    let prefix = format!("model.layers.0.experts.test_{bits}");
    let infos = [
        TensorInfo {
            name: format!("{prefix}.weight"),
            file: path.clone(),
            dtype: "U32".into(),
            shape: vec![MATRICES, OUTPUT, words_per_row],
            data_start: 0,
            data_offsets: [0, weight_end],
        },
        TensorInfo {
            name: format!("{prefix}.scales"),
            file: path.clone(),
            dtype: "BF16".into(),
            shape: vec![MATRICES, OUTPUT, 1],
            data_start: 0,
            data_offsets: [weight_end, scale_end],
        },
        TensorInfo {
            name: format!("{prefix}.biases"),
            file: path.clone(),
            dtype: "BF16".into(),
            shape: vec![MATRICES, OUTPUT, 1],
            data_start: 0,
            data_offsets: [scale_end, bias_end],
        },
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for info in &infos {
        upload.enqueue(info)?;
    }
    let tensors = upload.finish()?;
    let selected = AffineQuantizedTensors {
        weight: required(&tensors, &infos[0].name)?,
        scales: required(&tensors, &infos[1].name)?,
        biases: required(&tensors, &infos[2].name)?,
    };
    let mut host_input = backend.inner.context.allocate_pinned::<bf16>(INPUT)?;
    host_input.copy_from_slice(&[bf16::from_f32(1.0); INPUT])?;
    let mut input = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, INPUT)?;
    backend.inner.stream.copy_to_device(&mut host_input, &mut input)?;
    let linear = backend.prepare_affine_quantized_bf16_linear(INPUT, OUTPUT, MATRICES, 64, bits)?;
    let mut output = backend
        .inner
        .pool
        .allocate_zeroed::<bf16>(&backend.inner.stream, linear.output_elements())?;
    linear.execute(&input, selected, &mut output, 1)?;
    let mut host_output = backend.inner.context.allocate_pinned::<bf16>(OUTPUT)?;
    backend.inner.stream.copy_to_host(&output, &mut host_output)?;
    assert_eq!(host_output.to_vec()?, [bf16::from_f32(-32.0), bf16::from_f32(64.0)]);
    fs::remove_file(path)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn append_packed_row(bytes: &mut Vec<u8>, value: u32, bits: usize, values: usize) {
    let mut words = vec![0_u32; values * bits / 32];
    for index in 0..values {
        let bit = index * bits;
        words[bit / 32] |= value << (bit % 32);
        if bit % 32 + bits > 32 {
            words[bit / 32 + 1] |= value >> (32 - bit % 32);
        }
    }
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}

#[cfg(target_os = "linux")]
fn append_bf16<const N: usize>(bytes: &mut Vec<u8>, values: [f32; N]) {
    for value in values.map(bf16::from_f32) {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

#[cfg(target_os = "linux")]
fn required<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}

#[cfg(target_os = "linux")]
fn temp_path() -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-linear-{}.bin", std::process::id()))
}

#[cfg(target_os = "linux")]
fn temp_path_for(bits: usize) -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-affine-{bits}-{}.bin", std::process::id()))
}
