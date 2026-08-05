mod prepare;
pub(super) mod support;
mod tiled;

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

use super::*;
use crate::{CudaConfig, NvFp4ExpertBankConfig, NvFp4ExpertSource};

const BASE: &str = "model.language_model.layers.0.experts";
const EXPERTS: usize = 8;
const HIDDEN: usize = 2_816;
const INTERMEDIATE: usize = 704;

#[test]
fn checkpoint_selected_nvfp4_matches_w4a16_reference() -> Result<()> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let gate_sources = sources(&catalog, "gate_proj")?;
    let up_sources = sources(&catalog, "up_proj")?;
    let down_sources = sources(&catalog, "down_proj")?;
    let gate = backend.prepare_nvfp4_expert_bank(bank(HIDDEN, INTERMEDIATE), &gate_sources)?;
    let up = backend.prepare_nvfp4_expert_bank(bank(HIDDEN, INTERMEDIATE), &up_sources)?;
    let down = backend.prepare_nvfp4_expert_bank(bank(INTERMEDIATE, HIDDEN), &down_sources)?;
    let mut moe = backend.prepare_selected_nvfp4_moe_bf16(
        EXPERTS,
        GatedActivation::GeluTanh,
        gate,
        up,
        down,
    )?;
    let input_values = values(HIDDEN)?;
    let input = copy(&backend, &input_values)?;
    let selected = copy(&backend, &(0_u32..u32::try_from(EXPERTS)?).collect::<Vec<_>>())?;
    let routing = copy(&backend, &[bf16::from_f32(0.125); EXPERTS])?;
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    moe.execute(&input, &selected, &routing, &mut output)?;
    let intermediate = read(&backend, &moe.intermediate)?;
    let actual = read(&backend, &output)?;
    check_gated(&input_values, &intermediate, &gate_sources, &up_sources)?;
    check_down(&intermediate, &actual, &down_sources)
}

fn check_gated(
    input: &[bf16],
    actual: &[bf16],
    gate_sources: &[NvFp4ExpertSource<'_>],
    up_sources: &[NvFp4ExpertSource<'_>],
) -> Result<()> {
    for expert in 0..EXPERTS {
        let gate = RawProjection::read(gate_sources[expert])?;
        let up = RawProjection::read(up_sources[expert])?;
        for row in 0..8 {
            let gate_value = dot(input, &gate, row);
            let up_value = dot(input, &up, row);
            let cube = gate_value * gate_value * gate_value;
            let activated = 0.5
                * gate_value
                * (1.0 + (0.797_884_6 * 0.044_715_f32.mul_add(cube, gate_value)).tanh());
            close(actual[expert * INTERMEDIATE + row].to_f32(), activated * up_value);
        }
    }
    Ok(())
}

fn check_down(input: &[bf16], actual: &[bf16], sources: &[NvFp4ExpertSource<'_>]) -> Result<()> {
    let projections =
        sources.iter().copied().map(RawProjection::read).collect::<Result<Vec<_>>>()?;
    for (row, actual) in actual.iter().enumerate().take(8) {
        let expected = projections.iter().enumerate().fold(0.0_f32, |sum, (expert, weight)| {
            let start = expert * INTERMEDIATE;
            dot(&input[start..start + INTERMEDIATE], weight, row).mul_add(0.125, sum)
        });
        close(actual.to_f32(), expected);
    }
    Ok(())
}

fn close(actual: f32, expected: f32) {
    let tolerance = (expected.abs() * 0.03).max(0.5);
    assert!((actual - expected).abs() <= tolerance, "actual={actual} expected={expected}");
}

struct RawProjection {
    weight: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
    input: usize,
}

impl RawProjection {
    fn read(source: NvFp4ExpertSource<'_>) -> Result<Self> {
        Ok(Self {
            weight: payload(source.weight)?,
            scales: payload(source.weight_scale)?,
            global: scalar(source.weight_scale_2)?,
            input: source.weight.shape[1] * 2,
        })
    }
}

fn dot(input: &[bf16], projection: &RawProjection, row: usize) -> f32 {
    input.iter().enumerate().fold(0.0_f32, |sum, (column, value)| {
        value.to_f32().mul_add(dequant(projection, row, column), sum)
    })
}

fn dequant(projection: &RawProjection, row: usize, column: usize) -> f32 {
    let index = row * projection.input + column;
    let byte = projection.weight[index / 2];
    let packed = if index.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let scale = projection.scales[row * (projection.input / 16) + column / 16];
    fp4(packed) * fp8(scale) * projection.global
}

fn fp4(value: u8) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = VALUES[usize::from(value & 7)];
    if value & 8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn fp8(value: u8) -> f32 {
    let sign = if value & 0x80 == 0 {
        1.0
    } else {
        -1.0
    };
    let exponent = i32::from((value >> 3) & 0x0f);
    let mantissa = value & 7;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(exponent - 7)
    }
}

fn bank(input: usize, output: usize) -> NvFp4ExpertBankConfig {
    NvFp4ExpertBankConfig {
        experts: EXPERTS,
        input_features: input,
        output_features: output,
    }
}

fn sources<'a>(catalog: &'a TensorCatalog, projection: &str) -> Result<Vec<NvFp4ExpertSource<'a>>> {
    (0..EXPERTS)
        .map(|expert| {
            let prefix = format!("{BASE}.{expert}.{projection}");
            Ok(NvFp4ExpertSource {
                weight: required(catalog, &format!("{prefix}.weight"))?,
                weight_scale: required(catalog, &format!("{prefix}.weight_scale"))?,
                weight_scale_2: required(catalog, &format!("{prefix}.weight_scale_2"))?,
                input_scale: required(catalog, &format!("{prefix}.input_scale"))?,
            })
        })
        .collect()
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}

fn values(count: usize) -> Result<Vec<bf16>> {
    (0..count)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 16.0 - 0.9375)))
        .collect()
}

fn payload(info: &TensorInfo) -> Result<Vec<u8>> {
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(info.payload_start()?))?;
    let mut bytes = vec![0; info.payload_bytes()?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn scalar(info: &TensorInfo) -> Result<f32> {
    let bytes = payload(info)?;
    if bytes.len() != 4 {
        return Err(Error::InvalidNvFp4("invalid global scale"));
    }
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes);
    Ok(f32::from_le_bytes(value))
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
