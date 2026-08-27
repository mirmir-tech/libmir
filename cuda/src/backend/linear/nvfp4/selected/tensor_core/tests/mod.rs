mod moe;

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

use super::{super::tests::support as moe_test_support, *};
use crate::{CudaConfig, NvFp4ExpertBankConfig, NvFp4ExpertSource};

const BASE: &str = "model.language_model.layers.0.experts";
const EXPERTS: usize = 8;
const HIDDEN: usize = 2_816;
const INTERMEDIATE: usize = 704;

#[test]
fn checkpoint_selected_nvfp4_tensor_core_matches_reference() -> Result<()> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let sources = sources(&catalog)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let bank = backend.prepare_nvfp4_expert_bank(
        NvFp4ExpertBankConfig {
            experts: EXPERTS,
            input_features: HIDDEN,
            output_features: INTERMEDIATE,
        },
        &sources,
    )?;
    let mut linear = backend.prepare_selected_nvfp4_linear_bf16(EXPERTS, bank)?;
    let input_values = values(HIDDEN)?;
    let input = copy(&backend, &input_values)?;
    let selected_values = (0..u32::try_from(EXPERTS)?).collect::<Vec<_>>();
    let selected = copy(&backend, &selected_values)?;
    let mut output = backend
        .inner
        .pool
        .allocate::<bf16>(&backend.inner.stream, linear.output_elements()?)?;
    linear.execute(&input, &selected, &mut output)?;
    let actual = read(&backend, &output)?;
    for expert in 0..EXPERTS {
        let projection = RawProjection::read(sources[expert])?;
        for row in 0..8 {
            let expected = dot(&input_values, &projection, row);
            let value = actual[expert * INTERMEDIATE + row].to_f32();
            let tolerance = (expected.abs() * 0.08).max(1.0);
            assert!((value - expected).abs() <= tolerance, "expert={expert} row={row}");
        }
    }
    Ok(())
}

struct RawProjection {
    weight: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
}

impl RawProjection {
    fn read(source: NvFp4ExpertSource<'_>) -> Result<Self> {
        Ok(Self {
            weight: payload(source.weight)?,
            scales: payload(source.weight_scale)?,
            global: scalar(source.weight_scale_2)?,
        })
    }
}

fn dot(input: &[bf16], projection: &RawProjection, row: usize) -> f32 {
    input.iter().enumerate().fold(0.0_f32, |sum, (column, value)| {
        let index = row * HIDDEN + column;
        let byte = projection.weight[index / 2];
        let packed = if index.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        let scale = projection.scales[row * (HIDDEN / 16) + column / 16];
        value.to_f32().mul_add(fp4(packed) * fp8(scale) * projection.global, sum)
    })
}

fn fp4(value: u8) -> f32 {
    const VALUES: [f32; 16] =
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(value)]
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

fn sources(catalog: &TensorCatalog) -> Result<Vec<NvFp4ExpertSource<'_>>> {
    (0..EXPERTS)
        .map(|expert| {
            let prefix = format!("{BASE}.{expert}.gate_proj");
            Ok(NvFp4ExpertSource {
                weight: required(catalog, &format!("{prefix}.weight"))?,
                weight_scale: required(catalog, &format!("{prefix}.weight_scale"))?,
                weight_scale_2: required(catalog, &format!("{prefix}.weight_scale_2"))?,
                input_scale: required(catalog, &format!("{prefix}.input_scale"))?,
                scale_mode: crate::NvFp4ScaleMode::Multiplier,
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

fn values(count: usize) -> Result<Vec<bf16>> {
    (0..count)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 16.0 - 0.9375)))
        .collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
