use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use mircuda::{DeviceBuffer, DeviceElement};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

use super::*;
use crate::{
    CudaConfig, NvFp4ExpertBankConfig, NvFp4ExpertSource,
    kernels::{NvFp4SelectedWeightLaunch, NvFp4SelectedWeightPreparation, NvFp4Spec},
};

const BASE: &str = "model.language_model.layers.0.experts";
const EXPERTS: usize = 8;
const SELECTED: usize = 3;
const HIDDEN: usize = 2_816;
const INTERMEDIATE: usize = 704;

#[test]
fn checkpoint_selected_nvfp4_preparation_matches_source() -> Result<()> {
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
    let selected = copy(&backend, &[u32::try_from(SELECTED)?])?;
    let spec = NvFp4Spec::new(HIDDEN, INTERMEDIATE)?;
    let mut weight =
        backend.inner.pool.allocate::<u8>(&backend.inner.stream, spec.elements()? / 2)?;
    let mut scales = backend
        .inner
        .pool
        .allocate_zeroed::<u8>(&backend.inner.stream, spec.scale_elements()?)?;
    let mut input_scale = backend.inner.pool.allocate::<f32>(&backend.inner.stream, 1)?;
    let mut weight_scale = backend.inner.pool.allocate::<f32>(&backend.inner.stream, 1)?;
    let preparation = NvFp4SelectedWeightPreparation::compile(&backend.inner.compiler)?;
    preparation.execute(
        &backend.inner.stream,
        spec,
        EXPERTS,
        &mut NvFp4SelectedWeightLaunch {
            source_weight: &bank.weight,
            source_scales: &bank.scales,
            source_input_scales: &bank.input_scales,
            source_weight_scales: &bank.global_scales,
            selected: &selected,
            rank: 0,
            weight: &mut weight,
            scales: &mut scales,
            input_scale: &mut input_scale,
            weight_scale: &mut weight_scale,
        },
    )?;
    backend.synchronize()?;
    assert_eq!(read(&backend, &weight)?, payload(sources[SELECTED].weight)?);
    check_scales(&read(&backend, &scales)?, &payload(sources[SELECTED].weight_scale)?);
    assert_eq!(read(&backend, &input_scale)?, vec![scalar(sources[SELECTED].input_scale)?]);
    assert_eq!(read(&backend, &weight_scale)?, vec![scalar(sources[SELECTED].weight_scale_2)?]);
    Ok(())
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

fn check_scales(actual: &[u8], source: &[u8]) {
    let blocks = HIDDEN / 16;
    for (index, expected) in source.iter().enumerate() {
        let row = index / blocks;
        let block = index % blocks;
        assert_eq!(actual[scale_offset(row, block)], *expected, "scale {index}");
    }
}

fn scale_offset(row: usize, block: usize) -> usize {
    let tile = (row / 128) * (HIDDEN / 64) + block / 4;
    tile * 512 + (row % 32) * 16 + (row / 32 % 4) * 4 + block % 4
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
