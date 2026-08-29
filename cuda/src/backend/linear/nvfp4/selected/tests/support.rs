use std::path::Path;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

use super::*;
use crate::{MoeExecution, NvFp4ExpertBankConfig, NvFp4ExpertSource};

const BASE: &str = "model.language_model.layers.0.experts";
pub(in crate::backend::linear::nvfp4::selected) const EXPERTS: usize = 8;
pub(in crate::backend::linear::nvfp4::selected) const HIDDEN: usize = 2_816;
pub(in crate::backend::linear::nvfp4::selected) const INTERMEDIATE: usize = 704;

pub(in crate::backend::linear::nvfp4::selected) fn catalog() -> Result<Option<TensorCatalog>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(None);
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    Ok(Some(TensorCatalog::from_layout(&layout)?))
}

pub(in crate::backend::linear::nvfp4::selected) fn bank(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    projection: &str,
    input: usize,
    output: usize,
) -> Result<NvFp4ExpertBank> {
    let sources = sources(catalog, projection)?;
    backend.prepare_nvfp4_expert_bank(
        NvFp4ExpertBankConfig {
            experts: EXPERTS,
            input_features: input,
            output_features: output,
        },
        &sources,
    )
}

pub(in crate::backend::linear::nvfp4::selected) fn banks(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
) -> Result<(NvFp4ExpertBank, NvFp4ExpertBank, NvFp4ExpertBank)> {
    Ok((
        bank(backend, catalog, "gate_proj", HIDDEN, INTERMEDIATE)?,
        bank(backend, catalog, "up_proj", HIDDEN, INTERMEDIATE)?,
        bank(backend, catalog, "down_proj", INTERMEDIATE, HIDDEN)?,
    ))
}

pub(in crate::backend::linear::nvfp4::selected) fn with_banks<T>(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    prepare: impl FnOnce(NvFp4ExpertBank, NvFp4ExpertBank, NvFp4ExpertBank) -> Result<T>,
) -> Result<T> {
    let (gate, up, down) = banks(backend, catalog)?;
    prepare(gate, up, down)
}

pub(in crate::backend::linear::nvfp4::selected) struct MoeCandidates {
    pub reference: SelectedNvFp4MoeBf16,
    pub tensor_core: SelectedNvFp4TensorCoreMoeBf16,
    pub grouped: GroupedNvFp4MoeBf16,
    pub fused: GroupedNvFp4MoeBf16,
    pub direct: DirectNvFp4MoeBf16,
    pub direct_down: DirectDownNvFp4MoeBf16,
    pub hybrid: HybridNvFp4MoeBf16,
}

pub(in crate::backend::linear::nvfp4::selected) fn candidates(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
) -> Result<MoeCandidates> {
    let activation = GatedActivation::GeluTanh;
    Ok(MoeCandidates {
        reference: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_selected_nvfp4_moe_bf16(EXPERTS, activation, gate, up, down)
        })?,
        tensor_core: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_selected_nvfp4_tensor_core_moe_bf16(EXPERTS, activation, gate, up, down)
        })?,
        grouped: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_grouped_nvfp4_moe_bf16(
                1,
                EXPERTS,
                activation,
                MoeExecution::IndexedGrouped,
                gate,
                up,
                down,
            )
        })?,
        fused: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_grouped_nvfp4_moe_bf16(
                1,
                EXPERTS,
                activation,
                MoeExecution::FusedIndexedGrouped,
                gate,
                up,
                down,
            )
        })?,
        direct: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_direct_nvfp4_moe_bf16(1, EXPERTS, activation, gate, up, down)
        })?,
        direct_down: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_direct_down_nvfp4_moe_bf16(1, EXPERTS, activation, gate, up, down)
        })?,
        hybrid: with_banks(backend, catalog, |gate, up, down| {
            backend.prepare_hybrid_gate_nvfp4_moe_bf16(1, EXPERTS, activation, gate, up, down)
        })?,
    })
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

pub(in crate::backend::linear::nvfp4::selected) fn values(count: usize) -> Result<Vec<bf16>> {
    (0..count)
        .map(|index| Ok(bf16::from_f32(f32::from(u8::try_from(index % 31)?) / 16.0 - 0.9375)))
        .collect()
}

pub(in crate::backend::linear::nvfp4::selected) fn copy<T: DeviceElement>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

pub(in crate::backend::linear::nvfp4::selected) fn read<T: DeviceElement>(
    backend: &CudaBackend,
    source: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
