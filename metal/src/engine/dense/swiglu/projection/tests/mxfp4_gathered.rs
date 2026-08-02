use std::{fs, path::Path};

use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::engine::Dtype;

#[test]
fn gathers_pinned_mxfp4_matrix_bank() -> Result<()> {
    gather(BlockQuantization::MXFP4, "U8", &[2, 2, 1, 16], "u8")
}

#[test]
fn gathers_mlx_u32_mxfp4_matrix_bank() -> Result<()> {
    gather(BlockQuantization::MXFP4_MLX, "U32", &[2, 2, 4], "u32")
}

#[test]
fn broadcasts_mlx_mxfp4_input_across_expert_selections() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-mxfp4-broadcast-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"), "U32", &[2, 2, 4])?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let linear =
        BoundLinear::load(&tensors, &binding(BlockQuantization::MXFP4_MLX, &[2, 2, 4]), &stream)?;
    let input = Array::from_f32(&[1.0; 32], &[1, 1, 1, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[1, 0], &[1, 1, 2])?;

    let output = linear.gather(&input, &indices, false, &stream)?;
    assert_eq!(output.shape()?, [1, 1, 2, 1, 2]);
    assert_eq!(output.to_vec_f32_on_stream(&stream)?, [51.0, 68.0, 17.0, 34.0]);

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn gathers_and_splits_interleaved_mxfp4_gate_up_bank() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-mxfp4-interleaved-{}", std::process::id()));
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
    assert_eq!(gate.shape()?, [2, 1, 2]);
    assert_eq!(gate.to_vec_f32_on_stream(&stream)?, [16.0, 48.0, 96.0, 192.0]);
    assert_eq!(up.to_vec_f32_on_stream(&stream)?, [32.0, 64.0, 128.0, 16.0]);

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn gather(format: BlockQuantization, dtype: &str, shape: &[usize], label: &str) -> Result<()> {
    let root = std::env::temp_dir()
        .join(format!("libmir-metal-mxfp4-gathered-{label}-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"), dtype, shape)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &binding(format, shape), &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[2, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[1, 0], &[2])?;

    assert!(linear.forward(&input, &stream).is_err());
    let output = linear.gather(&input, &indices, false, &stream)?;
    assert_eq!(output.shape()?, [2, 1, 2]);
    assert_eq!(output.to_vec_f32_on_stream(&stream)?, [51.0, 68.0, 17.0, 34.0]);

    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding(format: BlockQuantization, shape: &[usize]) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: ExpertProjectionRole::Gate,
            },
        },
        source: "weight".into(),
        shape: shape.into(),
        logical_shape: Some(vec![2, 2, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: TensorStorage::BlockQuantized {
            format,
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
        shape: vec![2, 4, 1, 16],
        logical_shape: Some(vec![2, 4, 32]),
        transforms: vec![
            BindingTransform::StackedExperts { count: 2 },
            BindingTransform::FusedGateUp { interleaved: true },
        ],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: "scales".into(),
            global_scale: None,
            input_scale: None,
            bias: None,
            packing: TensorPacking::InterleavedGateUp,
        },
    }
}

fn write_interleaved_safetensors(path: &Path) -> Result<()> {
    let mut payload = [0x11_u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x11]
        .into_iter()
        .flat_map(|packed| [packed; 16])
        .collect::<Vec<_>>();
    let weight_end = payload.len();
    payload.extend([127_u8; 8]);
    let end = payload.len();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"U8","shape":[2,4,1,16],"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"U8","shape":[2,4,1],"data_offsets":[{weight_end},{end}]}}}}"#
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

fn write_safetensors(path: &Path, dtype: &str, shape: &[usize]) -> Result<()> {
    let mut payload = [0x11_u8; 16]
        .into_iter()
        .chain([0x22_u8; 16])
        .chain([0x33_u8; 16])
        .chain([0x44_u8; 16])
        .collect::<Vec<_>>();
    let weight_end = payload.len();
    payload.extend([127_u8; 4]);
    let scales_end = payload.len();
    for value in [0x3f80_u16, 0x4000, 0x4040, 0x4080] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let end = payload.len();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[0,{weight_end}]}},"scales":{{"dtype":"U8","shape":[2,2,1],"data_offsets":[{weight_end},{scales_end}]}},"bias":{{"dtype":"BF16","shape":[2,2],"data_offsets":[{scales_end},{end}]}}}}"#
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
