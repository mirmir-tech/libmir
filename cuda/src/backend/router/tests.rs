use std::path::Path;

use mircuda::{DeviceBuffer, bf16};
use models::{
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

use super::*;
use crate::{CudaConfig, CudaTensorSet};

const BASE: &str = "model.language_model.layers.0.router";

#[test]
fn checkpoint_router_matches_bf16_reference() -> Result<()> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let names = [
        format!("{BASE}.proj.weight"),
        format!("{BASE}.scale"),
        format!("{BASE}.per_expert_scale"),
    ];
    let infos = names.iter().map(|name| required(&catalog, name)).collect::<Result<Vec<_>>>()?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &infos)?;
    let input_values = input_values()?;
    let input = copy(&backend, &input_values)?;
    let spec = RouterSpec {
        hidden: 2_816,
        experts: 128,
        top_k: 8,
        epsilon: 1.0e-6,
        norm_multiplier: 1.0 / 2_816.0_f32.sqrt(),
    };
    let mut router = backend.prepare_router_bf16(spec)?;
    let selection = router.execute(
        &input,
        RouterTensors {
            projection: tensor(&tensors, &names[0])?,
            norm_scale: tensor(&tensors, &names[1])?,
            expert_scale: tensor(&tensors, &names[2])?,
        },
    )?;
    let actual_indices = read(&backend, selection.indices)?;
    let actual_weights = read(&backend, selection.weights)?;
    let projection = tensors.read_bf16(&names[0])?;
    let norm_scale = tensors.read_bf16(&names[1])?;
    let expert_scale = tensors.read_bf16(&names[2])?;
    let expected = reference(spec, &input_values, &projection, &norm_scale, &expert_scale)?;
    assert_eq!(actual_indices, expected.0);
    for (actual, expected) in actual_weights.iter().zip(expected.1) {
        assert!((actual.to_f32() - expected.to_f32()).abs() < 0.002);
    }
    Ok(())
}

fn reference(
    spec: RouterSpec,
    input: &[bf16],
    projection: &[bf16],
    norm_scale: &[bf16],
    expert_scale: &[bf16],
) -> Result<(Vec<u32>, Vec<bf16>)> {
    let sum = input
        .iter()
        .fold(0.0_f32, |sum, value| value.to_f32().mul_add(value.to_f32(), sum));
    let hidden = f32::from(u16::try_from(spec.hidden)?);
    let inverse = (sum / hidden + spec.epsilon).sqrt().recip();
    let normalized: Vec<bf16> = input
        .iter()
        .zip(norm_scale)
        .map(|(value, scale)| {
            let scale = bf16::from_f32(scale.to_f32() * spec.norm_multiplier).to_f32();
            bf16::from_f32(value.to_f32() * inverse * scale)
        })
        .collect();
    let mut scores: Vec<(usize, f32)> = projection
        .chunks_exact(spec.hidden)
        .enumerate()
        .map(|(expert, row)| {
            let score = normalized
                .iter()
                .zip(row)
                .fold(0.0_f32, |sum, (input, weight)| input.to_f32().mul_add(weight.to_f32(), sum));
            (expert, score)
        })
        .collect();
    scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    scores.truncate(spec.top_k);
    let maximum = scores[0].1;
    let denominator: f32 = scores.iter().map(|(_, score)| (*score - maximum).exp()).sum();
    let indices = scores
        .iter()
        .map(|(expert, _)| Ok(u32::try_from(*expert)?))
        .collect::<Result<Vec<_>>>()?;
    let weights = scores
        .iter()
        .map(|(expert, score)| {
            let probability = (*score - maximum).exp() / denominator;
            bf16::from_f32(probability * expert_scale[*expert].to_f32())
        })
        .collect();
    Ok((indices, weights))
}

fn input_values() -> Result<Vec<bf16>> {
    (0..2_816)
        .map(|index| {
            let value = f32::from(u8::try_from(index % 31)?) / 16.0 - 0.9375;
            Ok(bf16::from_f32(value))
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

fn upload(backend: &CudaBackend, infos: &[&TensorInfo]) -> Result<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    for info in infos {
        upload.enqueue(info)?;
    }
    upload.finish()
}

fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}

fn copy(backend: &CudaBackend, values: &[bf16]) -> Result<DeviceBuffer<bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}

fn read<T: mircuda::DeviceElement>(
    backend: &CudaBackend,
    source: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
