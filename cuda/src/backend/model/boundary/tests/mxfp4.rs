use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, TensorPacking, TensorStorage,
};

use super::*;

#[test]
fn executes_mxfp4_output_head_without_dense_weights() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-mxfp4-boundary-{}.bin", std::process::id()));
    let mut bytes = [0x22_u8; 16].into_iter().chain([0x33_u8; 16]).collect::<Vec<_>>();
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8, 128]);
    let scale_end = u64::try_from(bytes.len())?;
    for value in [bf16::ONE, bf16::from_f32(-2.0)] {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let end = u64::try_from(bytes.len())?;
    fs::write(&path, bytes)?;
    let infos = [
        info("boundary_blocks", &path, "U8", vec![2, 1, 16], 0, weight_end),
        info("boundary_scales", &path, "U8", vec![2, 1], weight_end, scale_end),
        info("boundary_bias", &path, "BF16", vec![2], scale_end, end),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let weight = CheckpointProjectionWeight::load_binding(&tensors, &binding(true))?;
    let mut output =
        ModelOutputHeadTemplate::prepare(&backend, weight, 32, 2)?.instantiate(&backend)?;
    let input = copy(&backend, &[bf16::ONE; 32])?;
    let mut logits = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 2)?;
    output.execute(&input, &mut logits, SamplingLogits::Full)?;
    assert_eq!(read(&backend, &logits)?, [33.0_f32, 94.0].map(bf16::from_f32));
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn loads_and_executes_gathered_mxfp4_binding() -> Result<()> {
    let path = std::env::temp_dir()
        .join(format!("libmir-cuda-mxfp4-gathered-binding-{}.bin", process_id()));
    let mut bytes = [0x11_u8; 16]
        .into_iter()
        .chain([0x22_u8; 16])
        .chain([0x33_u8; 16])
        .chain([0x44_u8; 16])
        .collect::<Vec<_>>();
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 4]);
    let scale_end = u64::try_from(bytes.len())?;
    for value in [1.0_f32, 2.0, 3.0, 4.0].map(bf16::from_f32) {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let end = u64::try_from(bytes.len())?;
    fs::write(&path, bytes)?;
    let infos = [
        info("bank_blocks", &path, "U8", vec![2, 2, 1, 16], 0, weight_end),
        info("bank_scales", &path, "U8", vec![2, 2, 1], weight_end, scale_end),
        info("bank_bias", &path, "BF16", vec![2, 2], scale_end, end),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let weight = crate::MxFp4CheckpointWeight::load_binding(&tensors, &gathered_binding())?;
    assert!(weight.prepare(&backend, 2).is_err());
    let operation = weight.prepare_gathered(&backend, 2)?;
    let input = copy(&backend, &[bf16::ONE; 64])?;
    let selected = copy(&backend, &[1_u32, 0])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 4)?;
    operation.execute(&input, &selected, &weight, &mut output)?;
    assert_eq!(read(&backend, &output)?, [51.0_f32, 68.0, 17.0, 34.0].map(bf16::from_f32));
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn loads_and_executes_mlx_u32_gathered_mxfp4_binding() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-mxfp4-gathered-u32-{}.bin", process_id()));
    let mut bytes = [0x11_u8; 16]
        .into_iter()
        .chain([0x22_u8; 16])
        .chain([0x33_u8; 16])
        .chain([0x44_u8; 16])
        .collect::<Vec<_>>();
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 4]);
    let scale_end = u64::try_from(bytes.len())?;
    for value in [1.0_f32, 2.0, 3.0, 4.0].map(bf16::from_f32) {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let end = u64::try_from(bytes.len())?;
    fs::write(&path, bytes)?;
    let infos = [
        info("bank_mlx", &path, "U32", vec![2, 2, 4], 0, weight_end),
        info("bank_mlx_scales", &path, "U8", vec![2, 2, 1], weight_end, scale_end),
        info("bank_mlx_bias", &path, "BF16", vec![2, 2], scale_end, end),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let weight = crate::MxFp4CheckpointWeight::load_binding(&tensors, &gathered_mlx_binding())?;
    let operation = weight.prepare_gathered(&backend, 2)?;
    let input = copy(&backend, &[bf16::ONE; 64])?;
    let selected = copy(&backend, &[1_u32, 0])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 4)?;
    operation.execute(&input, &selected, &weight, &mut output)?;
    assert_eq!(read(&backend, &output)?, [51.0_f32, 68.0, 17.0, 34.0].map(bf16::from_f32));
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn executes_selected_mxfp4_embedding_row() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-mxfp4-embedding-{}.bin", process_id()));
    let mut bytes = [0x22_u8; 16].into_iter().chain([0x33_u8; 16]).collect::<Vec<_>>();
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8, 128]);
    let end = u64::try_from(bytes.len())?;
    fs::write(&path, bytes)?;
    let infos = [
        info("boundary_blocks", &path, "U8", vec![2, 1, 16], 0, weight_end),
        info("boundary_scales", &path, "U8", vec![2, 1], weight_end, end),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let weight = CheckpointProjectionWeight::load_binding(&tensors, &binding(false))?;
    let embedding = ModelEmbeddingTemplate::new(weight, 2, 32, 2.0)?.instantiate(&backend)?;
    let selected = copy(&backend, &[1_u32])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 32)?;
    embedding.execute(&selected, 0, &mut output)?;
    assert_eq!(read(&backend, &output)?, [bf16::from_f32(6.0); 32]);
    fs::remove_file(path)?;
    Ok(())
}

fn binding(with_bias: bool) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "boundary_blocks".into(),
        shape: vec![2, 1, 16],
        logical_shape: Some(vec![2, 32]),
        transforms: Vec::new(),
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: "boundary_scales".into(),
            global_scale: None,
            input_scale: None,
            bias: with_bias.then(|| "boundary_bias".into()),
            packing: TensorPacking::Separate,
        },
    }
}

fn gathered_binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: ExpertProjectionRole::Gate,
            },
        },
        source: "bank_blocks".into(),
        shape: vec![2, 2, 1, 16],
        logical_shape: Some(vec![2, 2, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: "bank_scales".into(),
            global_scale: None,
            input_scale: None,
            bias: Some("bank_bias".into()),
            packing: TensorPacking::Separate,
        },
    }
}

fn gathered_mlx_binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: ExpertProjectionRole::Gate,
            },
        },
        source: "bank_mlx".into(),
        shape: vec![2, 2, 4],
        logical_shape: Some(vec![2, 2, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4_MLX,
            scales: "bank_mlx_scales".into(),
            global_scale: None,
            input_scale: None,
            bias: Some("bank_mlx_bias".into()),
            packing: TensorPacking::Separate,
        },
    }
}

fn process_id() -> u32 {
    std::process::id()
}
