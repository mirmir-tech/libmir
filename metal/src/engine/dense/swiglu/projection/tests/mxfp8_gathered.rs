use std::{fs, path::Path};

use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::engine::Dtype;

#[test]
fn gathers_typed_mxfp8_matrix_bank() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-mxfp8-gathered-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &binding(), &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[2, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[1, 0], &[2])?;

    assert!(linear.forward(&input, &stream).is_err());
    let output = linear.gather(&input, &indices, false, &stream)?;
    assert_eq!(output.shape()?, [2, 1, 2]);
    assert_eq!(output.to_vec_f32_on_stream(&stream)?, [67.0, -28.0, 33.0, 18.0]);

    let input = Array::from_f32(&[1.0; 32], &[1, 1, 1, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[1, 0], &[1, 1, 2])?;
    let output = linear.gather(&input, &indices, false, &stream)?;
    assert_eq!(output.shape()?, [1, 1, 2, 1, 2]);
    assert_eq!(output.to_vec_f32_on_stream(&stream)?, [67.0, -28.0, 33.0, 18.0]);

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn gathers_and_splits_interleaved_mxfp8_gate_up_bank() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-mxfp8-interleaved-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_interleaved_safetensors(&root.join("model.safetensors"))?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &interleaved_binding(), &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[2, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[0, 1], &[2])?;

    let output = linear.gather(&input, &indices, false, &stream)?;
    let (gate, up) = crate::engine::fused_gate_up::split_interleaved_last(&output, 2, &stream)?;
    assert_eq!(gate.to_vec_f32_on_stream(&stream)?, [32.0, 96.0, 192.0, 384.0]);
    assert_eq!(up.to_vec_f32_on_stream(&stream)?, [64.0, 128.0, 256.0, 16.0]);

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: ExpertProjectionRole::Gate,
            },
        },
        source: "weight".into(),
        shape: vec![2, 2, 8],
        logical_shape: Some(vec![2, 2, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP8,
            scales: "scales".into(),
            global_scale: None,
            input_scale: None,
            bias: Some("bias".into()),
            packing: TensorPacking::Separate,
        },
    }
}

fn interleaved_binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: ExpertProjectionRole::GateUp,
            },
        },
        source: "weight".into(),
        shape: vec![2, 4, 8],
        logical_shape: Some(vec![2, 4, 32]),
        transforms: vec![
            BindingTransform::StackedExperts { count: 2 },
            BindingTransform::FusedGateUp { interleaved: true },
        ],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP8,
            scales: "scales".into(),
            global_scale: None,
            input_scale: None,
            bias: None,
            packing: TensorPacking::InterleavedGateUp,
        },
    }
}

fn write_safetensors(path: &Path) -> Result<()> {
    let mut payload = Vec::new();
    for word in [0x3838_3838_u32, 0x3030_3030, 0x3838_3838, 0xb8b8_b8b8] {
        for _ in 0..8 {
            payload.extend_from_slice(&word.to_le_bytes());
        }
    }
    let weight_end = payload.len();
    payload.extend_from_slice(&[127_u8, 127, 128, 127]);
    let scales_end = payload.len();
    for value in [0x3f80_u16, 0x4000, 0x4040, 0x4080] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let end = payload.len();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"U32","shape":[2,2,8],"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"U8","shape":[2,2,1],"data_offsets":[{weight_end},{scales_end}]}},"bias":{{"dtype":"BF16","shape":[2,2],"data_offsets":[{scales_end},{end}]}}}}"#
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

fn write_interleaved_safetensors(path: &Path) -> Result<()> {
    let mut payload = Vec::new();
    for byte in [0x38_u8, 0x40, 0x44, 0x48, 0x4c, 0x50, 0x54, 0x30] {
        let word = u32::from_le_bytes([byte; 4]);
        for _ in 0..8 {
            payload.extend_from_slice(&word.to_le_bytes());
        }
    }
    let weight_end = payload.len();
    payload.extend_from_slice(&[127_u8; 8]);
    let end = payload.len();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"U32","shape":[2,4,8],"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"U8","shape":[2,4,1],"data_offsets":[{weight_end},{end}]}}}}"#
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
