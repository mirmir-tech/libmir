use models::weights::{
    BindingTransform, BlockQuantization, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    RoutedExpertBindings, TensorBinding, TensorPacking, TensorStorage,
};

use super::*;
use crate::{
    ExecutionPhase, GatedActivation, MxFp8ExpertWeights, MxFp8GatheredMoeBf16, PlanSource,
    backend::tuning::{MoeProfileExecution, MoeProfileRequest, MxFp8MoeStorage},
};

#[test]
fn executes_complete_gathered_mxfp8_moe() -> Result<()> {
    let path = fixture("separate");
    let mut bytes = Vec::new();
    let gate = append_bank(&mut bytes, 0x3030_3030)?;
    let up = append_bank(&mut bytes, 0x3838_3838)?;
    let down = append_bank(&mut bytes, 0x3030_3030)?;
    let down_bias = append_bias(&mut bytes, 64.0)?;
    fs::write(&path, &bytes)?;
    let infos = [
        info("gate8", &path, "U32", vec![2, 32, 8], gate.0, gate.1),
        info("gate8_scales", &path, "U8", vec![2, 32, 1], gate.1, gate.2),
        info("up8", &path, "U32", vec![2, 32, 8], up.0, up.1),
        info("up8_scales", &path, "U8", vec![2, 32, 1], up.1, up.2),
        info("down8", &path, "U32", vec![2, 32, 8], down.0, down.1),
        info("down8_scales", &path, "U8", vec![2, 32, 1], down.1, down.2),
        info("down8_bias", &path, "BF16", vec![2, 32], down_bias.0, down_bias.1),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let gate = binding("gate8", "gate8_scales", None, ExpertProjectionRole::Gate);
    let up = binding("up8", "up8_scales", None, ExpertProjectionRole::Up);
    let down = binding("down8", "down8_scales", Some("down8_bias"), ExpertProjectionRole::Down);
    execute(
        &backend,
        &tensors,
        RoutedExpertBindings::SeparateGateUp { gate: &gate, up: &up, down: &down },
        MxFp8MoeStorage::Separate,
        true,
        8256.0,
    )?;
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn executes_interleaved_gathered_mxfp8_moe() -> Result<()> {
    let path = fixture("interleaved");
    let mut bytes = Vec::new();
    let gate_up = append_interleaved_bank(&mut bytes)?;
    let down = append_bank(&mut bytes, 0x3030_3030)?;
    fs::write(&path, &bytes)?;
    let infos = [
        info("gate_up8", &path, "U32", vec![2, 64, 8], gate_up.0, gate_up.1),
        info("gate_up8_scales", &path, "U8", vec![2, 64, 1], gate_up.1, gate_up.2),
        info("down8", &path, "U32", vec![2, 32, 8], down.0, down.1),
        info("down8_scales", &path, "U8", vec![2, 32, 1], down.1, down.2),
    ];
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let gate_up = interleaved_binding();
    let down = binding("down8", "down8_scales", None, ExpertProjectionRole::Down);
    execute(
        &backend,
        &tensors,
        RoutedExpertBindings::InterleavedGateUp { gate_up: &gate_up, down: &down },
        MxFp8MoeStorage::Interleaved,
        false,
        8192.0,
    )?;
    fs::remove_file(path)?;
    Ok(())
}

fn execute(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    bindings: RoutedExpertBindings<'_>,
    storage: MxFp8MoeStorage,
    has_bias: bool,
    expected: f32,
) -> Result<()> {
    let weights = MxFp8ExpertWeights::load(tensors, bindings, 2, 32, 32)?;
    let mut operation = MxFp8GatheredMoeBf16::new(backend, 1, 2, GatedActivation::Silu, &weights)?;
    let request = MoeProfileRequest::mxfp8(
        ExecutionPhase::Decode,
        1,
        2,
        2,
        32,
        32,
        storage,
        has_bias,
        GatedActivation::Silu,
    );
    assert!(matches!(
        backend.auto_tuner().lookup_moe(request),
        Some((MoeProfileExecution::MxFp8(_), PlanSource::MeasuredStartup))
    ));
    let input = copy(backend, &[bf16::ONE; 32])?;
    let selected = copy(backend, &[0_u32, 1])?;
    let routing = copy(backend, &[bf16::from_f32(0.25), bf16::from_f32(0.75)])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 32)?;
    operation.execute(&input, &selected, &routing, &weights, &mut output)?;
    assert_eq!(read(backend, &output)?, [bf16::from_f32(expected); 32]);
    Ok(())
}

fn append_bank(bytes: &mut Vec<u8>, packed: u32) -> Result<(u64, u64, u64)> {
    let start = u64::try_from(bytes.len())?;
    for _ in 0..2 * 32 * 8 {
        bytes.extend(packed.to_le_bytes());
    }
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 2 * 32]);
    Ok((start, weight_end, u64::try_from(bytes.len())?))
}

fn append_interleaved_bank(bytes: &mut Vec<u8>) -> Result<(u64, u64, u64)> {
    let start = u64::try_from(bytes.len())?;
    for _ in 0..2 * 32 {
        for packed in [0x3030_3030_u32, 0x3838_3838] {
            for _ in 0..8 {
                bytes.extend(packed.to_le_bytes());
            }
        }
    }
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 2 * 64]);
    Ok((start, weight_end, u64::try_from(bytes.len())?))
}

fn append_bias(bytes: &mut Vec<u8>, value: f32) -> Result<(u64, u64)> {
    let start = u64::try_from(bytes.len())?;
    for _ in 0..2 * 32 {
        bytes.extend(bf16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok((start, u64::try_from(bytes.len())?))
}

fn fixture(kind: &str) -> PathBuf {
    std::env::temp_dir().join(format!("libmir-cuda-mxfp8-{kind}-moe-{}.bin", std::process::id()))
}

fn interleaved_binding() -> TensorBinding {
    TensorBinding {
        role: role(ExpertProjectionRole::GateUp),
        source: "gate_up8".into(),
        shape: vec![2, 64, 8],
        logical_shape: Some(vec![2, 64, 32]),
        transforms: vec![
            BindingTransform::StackedExperts { count: 2 },
            BindingTransform::FusedGateUp { interleaved: true },
        ],
        storage: storage("gate_up8_scales", None, TensorPacking::InterleavedGateUp),
    }
}

fn binding(
    source: &str,
    scales: &str,
    bias: Option<&str>,
    projection: ExpertProjectionRole,
) -> TensorBinding {
    TensorBinding {
        role: role(projection),
        source: source.into(),
        shape: vec![2, 32, 8],
        logical_shape: Some(vec![2, 32, 32]),
        transforms: vec![BindingTransform::StackedExperts { count: 2 }],
        storage: storage(scales, bias, TensorPacking::Separate),
    }
}

const fn role(projection: ExpertProjectionRole) -> LogicalTensorRole {
    LogicalTensorRole::Layer {
        index: 0,
        tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
    }
}

fn storage(scales: &str, bias: Option<&str>, packing: TensorPacking) -> TensorStorage {
    TensorStorage::BlockQuantized {
        format: BlockQuantization::MXFP8,
        scales: scales.into(),
        global_scale: None,
        input_scale: None,
        bias: bias.map(Into::into),
        packing,
    }
}
