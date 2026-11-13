use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    RoutedExpertBindings, TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::{
    ExecutionPhase, GatedActivation, MxFp4ExpertWeights, MxFp4GatheredMoeBf16, PlanSource,
    backend::tuning::{MoeProfileExecution, MoeProfileRequest, MxFp4MoeStorage},
};

#[test]
fn executes_complete_gathered_mxfp4_moe() -> Result<()> {
    let path = std::env::temp_dir()
        .join(format!("libmir-cuda-mxfp4-gathered-moe-{}.bin", std::process::id()));
    let mut bytes = Vec::new();
    let gate = append_bank(&mut bytes, 0x11)?;
    let up = append_bank(&mut bytes, 0x22)?;
    let down = append_bank(&mut bytes, 0x11)?;
    fs::write(&path, &bytes)?;
    let infos = [
        info("gate", &path, "U8", vec![2, 32, 1, 16], gate.0, gate.1),
        info("gate_scales", &path, "U8", vec![2, 32, 1], gate.1, gate.2),
        info("up", &path, "U8", vec![2, 32, 1, 16], up.0, up.1),
        info("up_scales", &path, "U8", vec![2, 32, 1], up.1, up.2),
        info("down", &path, "U8", vec![2, 32, 1, 16], down.0, down.1),
        info("down_scales", &path, "U8", vec![2, 32, 1], down.1, down.2),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let gate = binding("gate", "gate_scales", ExpertProjectionRole::Gate);
    let up = binding("up", "up_scales", ExpertProjectionRole::Up);
    let down = binding("down", "down_scales", ExpertProjectionRole::Down);
    let weights = MxFp4ExpertWeights::load(
        &tensors,
        RoutedExpertBindings::SeparateGateUp { gate: &gate, up: &up, down: &down },
        2,
        32,
        32,
    )?;
    let mut operation = MxFp4GatheredMoeBf16::new(&backend, 1, 2, GatedActivation::Silu, &weights)?;
    let request = MoeProfileRequest::mxfp4(
        ExecutionPhase::Decode,
        1,
        2,
        2,
        32,
        32,
        MxFp4MoeStorage::Separate,
        GatedActivation::Silu,
    );
    assert!(matches!(
        backend.auto_tuner().lookup_moe(request),
        Some((MoeProfileExecution::MxFp4(_), PlanSource::MeasuredStartup))
    ));
    let input = copy(&backend, &[bf16::ONE; 32])?;
    let selected = copy(&backend, &[0_u32, 1])?;
    let routing = copy(&backend, &[bf16::from_f32(0.25), bf16::from_f32(0.75)])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 32)?;
    operation.execute(&input, &selected, &routing, &weights, &mut output)?;
    assert_eq!(read(&backend, &output)?, [bf16::from_f32(8192.0); 32]);
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn executes_interleaved_gathered_mxfp4_moe() -> Result<()> {
    let path = std::env::temp_dir()
        .join(format!("libmir-cuda-mxfp4-interleaved-moe-{}.bin", std::process::id()));
    let mut bytes = Vec::new();
    let gate_up = append_interleaved_bank(&mut bytes)?;
    let down = append_bank(&mut bytes, 0x11)?;
    fs::write(&path, &bytes)?;
    let infos = [
        info("gate_up", &path, "U8", vec![2, 64, 1, 16], gate_up.0, gate_up.1),
        info("gate_up_scales", &path, "U8", vec![2, 64, 1], gate_up.1, gate_up.2),
        info("down", &path, "U8", vec![2, 32, 1, 16], down.0, down.1),
        info("down_scales", &path, "U8", vec![2, 32, 1], down.1, down.2),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let gate_up = interleaved_binding();
    let down = binding("down", "down_scales", ExpertProjectionRole::Down);
    let weights = MxFp4ExpertWeights::load(
        &tensors,
        RoutedExpertBindings::InterleavedGateUp { gate_up: &gate_up, down: &down },
        2,
        32,
        32,
    )?;
    let mut operation = MxFp4GatheredMoeBf16::new(&backend, 1, 2, GatedActivation::Silu, &weights)?;
    let request = MoeProfileRequest::mxfp4(
        ExecutionPhase::Decode,
        1,
        2,
        2,
        32,
        32,
        MxFp4MoeStorage::Interleaved,
        GatedActivation::Silu,
    );
    assert!(matches!(
        backend.auto_tuner().lookup_moe(request),
        Some((MoeProfileExecution::MxFp4(_), PlanSource::MeasuredStartup))
    ));
    let input = copy(&backend, &[bf16::ONE; 32])?;
    let selected = copy(&backend, &[0_u32, 1])?;
    let routing = copy(&backend, &[bf16::from_f32(0.25), bf16::from_f32(0.75)])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 32)?;
    operation.execute(&input, &selected, &routing, &weights, &mut output)?;
    assert_eq!(read(&backend, &output)?, [bf16::from_f32(8192.0); 32]);
    fs::remove_file(path)?;
    Ok(())
}

fn append_bank(bytes: &mut Vec<u8>, packed: u8) -> Result<(u64, u64, u64)> {
    let start = u64::try_from(bytes.len())?;
    bytes.extend(std::iter::repeat_n(packed, 2 * 32 * 16));
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 2 * 32]);
    let scale_end = u64::try_from(bytes.len())?;
    Ok((start, weight_end, scale_end))
}

fn append_interleaved_bank(bytes: &mut Vec<u8>) -> Result<(u64, u64, u64)> {
    let start = u64::try_from(bytes.len())?;
    for _ in 0..2 * 32 {
        bytes.extend([0x11_u8; 16]);
        bytes.extend([0x22_u8; 16]);
    }
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 2 * 64]);
    let scale_end = u64::try_from(bytes.len())?;
    Ok((start, weight_end, scale_end))
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
        source: "gate_up".into(),
        shape: vec![2, 64, 1, 16],
        logical_shape: Some(vec![2, 64, 32]),
        transforms: vec![
            BindingTransform::StackedExperts { count: 2 },
            BindingTransform::FusedGateUp { interleaved: true },
        ],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: "gate_up_scales".into(),
            global_scale: None,
            input_scale: None,
            bias: None,
            packing: TensorPacking::InterleavedGateUp,
        },
    }
}

fn binding(source: &str, scales: &str, projection: ExpertProjectionRole) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
        },
        source: source.into(),
        shape: vec![2, 32, 1, 16],
        logical_shape: Some(vec![2, 32, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: scales.into(),
            global_scale: None,
            input_scale: None,
            bias: None,
            packing: TensorPacking::Separate,
        },
    }
}
