use std::fs;

use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, TensorPacking, TensorStorage,
};

use super::*;

#[test]
fn converts_nvfp4_once_on_metal_and_executes_dense_projection() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-nvfp4-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_fixture(&root.join("model.safetensors"))?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &binding(), &stream)?;
    let input =
        Array::from_f32(&[1.0; 16], &[1, 16])?.astype(crate::engine::Dtype::Bfloat16, &stream)?;
    assert_eq!(linear.forward(&input, &stream)?.to_vec_f32(&stream)?, [8.0, 6.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn gathers_nvfp4_matrix_bank_without_requantization() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-nvfp4-bank-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_bank_fixture(&root.join("model.safetensors"))?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &bank_binding(), &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[2, 1, 32])?
        .astype(crate::engine::Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[0, 1], &[2])?;
    let output = linear.gather(&input, &indices, false, &stream)?.to_vec_f32(&stream)?;

    assert_eq!(output, [17.0, 17.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn gathers_and_splits_interleaved_nvfp4_gate_up_bank() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-nvfp4-gate-up-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_interleaved_fixture(&root.join("model.safetensors"))?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let linear = BoundLinear::load(&tensors, &interleaved_binding(), &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[2, 1, 32])?
        .astype(crate::engine::Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[0, 1], &[2])?;

    let output = linear.gather(&input, &indices, false, &stream)?;
    let (gate, up) = crate::engine::fused_gate_up::split_interleaved_last(&output, 2, &stream)?;
    assert_eq!(gate.to_vec_f32(&stream)?, [16.0, 48.0, 96.0, 192.0]);
    assert_eq!(up.to_vec_f32(&stream)?, [32.0, 64.0, 128.0, 16.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn composes_individual_nvfp4_experts_on_device() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-nvfp4-individual-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_individual_fixture(&root.join("model.safetensors"))?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let bindings = [individual_binding(0), individual_binding(1)];
    let refs = bindings.iter().collect::<Vec<_>>();
    let linear = BoundLinear::load_nvfp4_bank(&tensors, &refs, &stream)?;
    let input = Array::from_f32(&[1.0; 64], &[2, 1, 32])?
        .astype(crate::engine::Dtype::Bfloat16, &stream)?;
    let indices = Array::from_u32(&[0, 1], &[2])?;
    assert_eq!(
        linear.gather(&input, &indices, false, &stream)?.to_vec_f32(&stream)?,
        [32.0, 64.0]
    );
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: vec![2, 8],
        logical_shape: Some(vec![2, 16]),
        transforms: Vec::new(),
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::NVFP4,
            scales: "weight_scale".into(),
            global_scale: Some("weight_scale_2".into()),
            input_scale: Some("input_scale".into()),
            bias: None,
            packing: TensorPacking::Separate,
        },
    }
}

fn bank_binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: ExpertProjectionRole::Gate,
            },
        },
        source: "weight".into(),
        shape: vec![2, 1, 16],
        logical_shape: Some(vec![2, 1, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::NVFP4,
            scales: "weight_scale".into(),
            global_scale: Some("weight_scale_2".into()),
            input_scale: Some("input_scale".into()),
            bias: None,
            packing: TensorPacking::Separate,
        },
    }
}

fn interleaved_binding() -> TensorBinding {
    let mut binding = bank_binding();
    binding.shape = vec![2, 4, 16];
    binding.logical_shape = Some(vec![2, 4, 32]);
    binding.transforms.push(BindingTransform::FusedGateUp { interleaved: true });
    binding.storage = TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales: "weight_scale".into(),
        global_scale: Some("weight_scale_2".into()),
        input_scale: Some("input_scale".into()),
        bias: None,
        packing: TensorPacking::InterleavedGateUp,
    };
    binding
}

fn individual_binding(expert: usize) -> TensorBinding {
    let mut binding = bank_binding();
    binding.role = LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::ExpertProjection {
            expert: Some(expert),
            projection: ExpertProjectionRole::Gate,
        },
    };
    binding.source = format!("weight{expert}");
    binding.shape = vec![1, 16];
    binding.logical_shape = Some(vec![1, 32]);
    binding.transforms.clear();
    binding.storage = TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales: format!("scale{expert}"),
        global_scale: Some(format!("global{expert}")),
        input_scale: Some(format!("input{expert}")),
        bias: None,
        packing: TensorPacking::Separate,
    };
    binding
}

fn write_fixture(path: &Path) -> Result<()> {
    let mut payload = vec![0x22_u8; 8];
    payload.extend([0xa4, 0x01].into_iter().cycle().take(8));
    payload.extend([0x38, 0x40]);
    payload.extend_from_slice(&0.5_f32.to_le_bytes());
    payload.extend_from_slice(&3.0_f32.to_le_bytes());
    let mut header = r#"{"weight":{"dtype":"U8","shape":[2,8],"data_offsets":[0,16]},"weight_scale":{"dtype":"F8_E4M3","shape":[2,1],"data_offsets":[16,18]},"weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[18,22]},"input_scale":{"dtype":"F32","shape":[],"data_offsets":[22,26]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}

fn write_bank_fixture(path: &Path) -> Result<()> {
    let mut payload = vec![0x22_u8; 32];
    payload.extend([0x38, 0x18, 0x38, 0x18]);
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    let mut header = r#"{"weight":{"dtype":"U8","shape":[2,1,16],"data_offsets":[0,32]},"weight_scale":{"dtype":"F8_E4M3","shape":[2,1,2],"data_offsets":[32,36]},"weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[36,40]},"input_scale":{"dtype":"F32","shape":[],"data_offsets":[40,44]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}

fn write_interleaved_fixture(path: &Path) -> Result<()> {
    let mut payload = [0x11_u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x11]
        .into_iter()
        .flat_map(|packed| [packed; 16])
        .collect::<Vec<_>>();
    payload.extend([0x38_u8; 16]);
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    let mut header = r#"{"weight":{"dtype":"U8","shape":[2,4,16],"data_offsets":[0,128]},"weight_scale":{"dtype":"F8_E4M3","shape":[2,4,2],"data_offsets":[128,144]},"weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[144,148]},"input_scale":{"dtype":"F32","shape":[],"data_offsets":[148,152]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}

fn write_individual_fixture(path: &Path) -> Result<()> {
    let mut payload = vec![0x22_u8; 16];
    payload.extend([0x38_u8; 2]);
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    payload.extend([0x22_u8; 16]);
    payload.extend([0x38_u8; 2]);
    payload.extend_from_slice(&2.0_f32.to_le_bytes());
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    let mut header = r#"{"weight0":{"dtype":"U8","shape":[1,16],"data_offsets":[0,16]},"scale0":{"dtype":"F8_E4M3","shape":[1,2],"data_offsets":[16,18]},"global0":{"dtype":"F32","shape":[],"data_offsets":[18,22]},"input0":{"dtype":"F32","shape":[],"data_offsets":[22,26]},"weight1":{"dtype":"U8","shape":[1,16],"data_offsets":[26,42]},"scale1":{"dtype":"F8_E4M3","shape":[1,2],"data_offsets":[42,44]},"global1":{"dtype":"F32","shape":[],"data_offsets":[44,48]},"input1":{"dtype":"F32","shape":[],"data_offsets":[48,52]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&payload);
    fs::write(path, data)?;
    Ok(())
}
