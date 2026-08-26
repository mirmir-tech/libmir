use std::{fs, path::Path};

use models::weights::{
    AffineBits, AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness,
    AffineStorageDType, AffineZeroPointMode, GroupedAffineQuantization, LogicalTensorRole,
    TensorBinding, TensorStorage,
};

use super::*;
use crate::engine::{Array, ModelTensors, Result, Stream};

mod awq;
mod bitsandbytes;
mod float8;
mod gptq;
mod mxfp4;
mod mxfp4_gathered;
mod mxfp8;
mod mxfp8_gathered;
mod nvfp4;
mod packed_integer;

#[test]
fn loads_dense_linear_and_tied_embedding_from_binding() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-bound-dense-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &stream)?;
    let binding = TensorBinding {
        role: LogicalTensorRole::Embedding,
        source: "weight".into(),
        shape: vec![2, 2],
        logical_shape: Some(vec![2, 2]),
        transforms: Vec::new(),
        storage: TensorStorage::Dense { dtype: "F32".into(), bias: None },
    };

    let linear = BoundLinear::load(&tensors, &binding, &stream)?;
    let input = Array::from_f32(&[1.0, 2.0], &[1, 2])?;
    assert_eq!(linear.forward(&input, &stream)?.to_vec_f32(&stream)?, [5.0, 11.0]);
    assert!(!linear.has_bias());

    let embedding = BoundEmbedding::load(&tensors, &binding, &stream)?;
    let indices = Array::from_u32(&[1], &[1])?;
    assert_eq!(embedding.lookup(&indices, &stream)?.to_vec_f32(&stream)?, [3.0, 4.0]);
    assert_eq!(embedding.project(&input, &stream)?.to_vec_f32(&stream)?, [5.0, 11.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn loads_every_native_affine_embedding_and_output_binding() -> Result<()> {
    for bits in [
        AffineBits::Two,
        AffineBits::Three,
        AffineBits::Four,
        AffineBits::Five,
        AffineBits::Six,
        AffineBits::Eight,
    ] {
        check_affine_roles(bits)?;
    }
    Ok(())
}

fn check_affine_roles(bits: AffineBits) -> Result<()> {
    let width = usize::from(bits.get());
    let root = std::env::temp_dir()
        .join(format!("libmir-metal-bound-affine-{width}-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_affine_safetensors(&root.join("model.safetensors"), width)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let embedding = BoundEmbedding::load(
        &tensors,
        &affine_binding(LogicalTensorRole::Embedding, bits),
        &stream,
    )?;
    let selected = Array::from_u32(&[1], &[1])?;
    assert_eq!(embedding.lookup(&selected, &stream)?.to_vec_f32(&stream)?, [2.0; 64]);
    let output =
        BoundLinear::load(&tensors, &affine_binding(LogicalTensorRole::Output, bits), &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[1, 64])?;
    assert_eq!(output.forward(&input, &stream)?.to_vec_f32(&stream)?, [64.0, 128.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn affine_binding(role: LogicalTensorRole, bits: AffineBits) -> TensorBinding {
    TensorBinding {
        role,
        source: "weight".into(),
        shape: vec![2, 64 * usize::from(bits.get()) / 32],
        logical_shape: Some(vec![2, 64]),
        transforms: Vec::new(),
        storage: TensorStorage::AffineQuantized {
            format: GroupedAffineQuantization {
                bits,
                group_size: 64,
                group_axis: AffineGroupAxis::Input,
                signedness: AffineSignedness::Unsigned,
                zero_point: AffineZeroPointMode::AdditiveBias,
                packing: AffinePacking::Mlx,
                storage_dtype: AffineStorageDType::U32,
                scale_dtype: AffineParameterDType::F32,
                bias_dtype: Some(AffineParameterDType::F32),
            },
            scales: "scales".into(),
            biases: Some("biases".into()),
            output_bias: None,
        },
    }
}

fn write_affine_safetensors(path: &Path, bits: usize) -> Result<()> {
    let mut payload = Vec::new();
    append_packed_row(&mut payload, 1, bits);
    append_packed_row(&mut payload, 2, bits);
    let weight_end = payload.len();
    for value in [1.0_f32; 2] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let scale_end = payload.len();
    payload.extend_from_slice(&[0_u8; 8]);
    let bias_end = payload.len();
    let words = 64 * bits / 32;
    let mut header = format!(
        r#"{{"weight":{{"dtype":"U32","shape":[2,{words}],"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"F32","shape":[2,1],"data_offsets":[{weight_end},{scale_end}]}},"biases":{{"dtype":"F32","shape":[2,1],"data_offsets":[{scale_end},{bias_end}]}}}}"#
    );
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}

fn append_packed_row(bytes: &mut Vec<u8>, value: u32, bits: usize) {
    let mut words = vec![0_u32; 64 * bits / 32];
    for index in 0..64 {
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

fn write_safetensors(path: &Path) -> Result<()> {
    let mut header = r#"{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    for value in [1.0_f32, 2.0, 3.0, 4.0] {
        data.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, data)?;
    Ok(())
}
