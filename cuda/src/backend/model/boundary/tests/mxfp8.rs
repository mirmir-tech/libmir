use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::MxFp8CheckpointWeight;

#[test]
fn loads_and_executes_gathered_mxfp8_bank() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("libmir-cuda-mxfp8-gathered-{}.bin", std::process::id()));
    let mut bytes = Vec::new();
    for word in [0x3838_3838_u32; 8]
        .into_iter()
        .chain([0xb0b0_b0b0; 8])
        .chain([0x3838_3838; 8])
        .chain([0xb0b0_b0b0; 8])
    {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8, 127, 128, 128]);
    let end = u64::try_from(bytes.len())?;
    fs::write(&path, bytes)?;
    let infos = [
        info("weight", &path, "U32", vec![2, 2, 8], 0, weight_end),
        info("scales", &path, "U8", vec![2, 2, 1], weight_end, end),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let weight = MxFp8CheckpointWeight::load_binding(&tensors, &binding())?;
    let operation = weight.prepare_gathered_routed(&backend, 1, 2)?;
    let input = copy(&backend, &[bf16::ONE; 32])?;
    let selected = copy(&backend, &[1_u32, 0])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 4)?;

    operation.execute(&input, &selected, &weight, &mut output)?;
    assert_eq!(
        read(&backend, &output)?,
        [bf16::from_f32(64.0), bf16::from_f32(-32.0), bf16::from_f32(32.0), bf16::from_f32(-16.0),]
    );
    fs::remove_file(path)?;
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
            bias: None,
            packing: TensorPacking::Separate,
        },
    }
}
