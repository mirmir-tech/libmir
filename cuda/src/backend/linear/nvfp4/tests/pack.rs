use std::path::Path;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::{layout::ModelLayout, weights::TensorCatalog};

use super::*;
use crate::{CudaConfig, CudaTensorSet};

const GATE: &str = "model.layers.0.mlp.gate_proj.weight";
const UP: &str = "model.layers.0.mlp.up_proj.weight";
const QUERY: &str = "model.layers.0.self_attn.q_proj.weight";
const KEY: &str = "model.layers.0.self_attn.k_proj.weight";
const VALUE: &str = "model.layers.0.self_attn.v_proj.weight";

#[test]
fn packed_checkpoint_pair_matches_separate_projections() -> Result<()> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_DENSE_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = upload(&backend, &catalog)?;
    let gate = weight(&backend, &tensors, GATE)?;
    let up = weight(&backend, &tensors, UP)?;
    let query = weight(&backend, &tensors, QUERY)?;
    let key = weight(&backend, &tensors, KEY)?;
    let value = weight(&backend, &tensors, VALUE)?;
    for tokens in [1, 2] {
        compare(&backend, tokens, &gate, &up)?;
        compare_qkv(&backend, tokens, &query, &key, &value)?;
    }
    Ok(())
}

fn compare_qkv(
    backend: &CudaBackend,
    tokens: usize,
    query: &NvFp4LinearWeight,
    key: &NvFp4LinearWeight,
    value: &NvFp4LinearWeight,
) -> Result<()> {
    let input_features = query.config().input_features;
    let widths = [
        query.config().output_features,
        key.config().output_features,
        value.config().output_features,
    ];
    let input = copy_device(
        backend,
        &(0..tokens * input_features).map(patterned_value).collect::<Vec<_>>(),
    )?;
    let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
    let mut separate = [
        allocate(tokens * widths[0])?,
        allocate(tokens * widths[1])?,
        allocate(tokens * widths[2])?,
    ];
    let mut group =
        NvFp4Bf16Pack::from_weights(backend, tokens, [query.clone(), key.clone(), value.clone()])?;
    group.execute(&input, &mut separate)?;
    let packed = backend.pack_nvfp4_linear_weights([query, key, value])?;
    let mut fused = NvFp4Bf16Linear::from_weight(backend, tokens, packed)?;
    let packed_width = widths.iter().sum::<usize>();
    let mut actual = allocate(tokens * packed_width)?;
    fused.execute(&input, &mut actual)?;
    let actual = read_device(backend, &actual)?;
    let expected = separate
        .iter()
        .map(|output| read_device(backend, output))
        .collect::<Result<Vec<_>>>()?;
    for row in 0..tokens {
        let mut offset = row * packed_width;
        for (width, output) in widths.into_iter().zip(&expected) {
            assert_eq!(&actual[offset..offset + width], &output[row * width..][..width]);
            offset += width;
        }
    }
    Ok(())
}

fn compare(
    backend: &CudaBackend,
    tokens: usize,
    gate: &NvFp4LinearWeight,
    up: &NvFp4LinearWeight,
) -> Result<()> {
    let input_features = gate.config().input_features;
    let output_features = gate.config().output_features;
    let input = copy_device(
        backend,
        &(0..tokens * input_features).map(patterned_value).collect::<Vec<_>>(),
    )?;
    let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
    let mut separate = [allocate(tokens * output_features)?, allocate(tokens * output_features)?];
    let mut pair = NvFp4Bf16Pack::from_weights(backend, tokens, [gate.clone(), up.clone()])?;
    pair.execute(&input, &mut separate)?;
    let packed = backend.pack_nvfp4_linear_weights([gate, up])?;
    let mut fused = NvFp4Bf16Linear::from_weight(backend, tokens, packed)?;
    let mut actual = allocate(tokens * output_features * 2)?;
    fused.execute(&input, &mut actual)?;
    let actual = read_device(backend, &actual)?;
    let gate = read_device(backend, &separate[0])?;
    let up = read_device(backend, &separate[1])?;
    for row in 0..tokens {
        let start = row * output_features * 2;
        assert_eq!(
            &actual[start..start + output_features],
            &gate[row * output_features..][..output_features]
        );
        assert_eq!(
            &actual[start + output_features..start + output_features * 2],
            &up[row * output_features..][..output_features]
        );
    }
    Ok(())
}

fn patterned_value(index: usize) -> bf16 {
    let value = u8::try_from(index % 31).unwrap_or_default();
    bf16::from_f32(f32::from(value) / 16.0 - 0.9375)
}

fn weight(backend: &CudaBackend, tensors: &CudaTensorSet, name: &str) -> Result<NvFp4LinearWeight> {
    let tensor = get(tensors, name)?;
    let base = name.strip_suffix(".weight").ok_or(Error::InvalidNvFp4("invalid weight name"))?;
    backend.prepare_nvfp4_linear_weight(
        NvFp4Config::new(tensor.shape()[1] * 2, tensor.shape()[0]),
        NvFp4Tensors {
            weight: tensor,
            weight_scale: get(tensors, &format!("{base}.weight_scale"))?,
            weight_scale_2: get(tensors, &format!("{base}.weight_scale_2"))?,
            input_scale: get(tensors, &format!("{base}.input_scale"))?,
            scale_mode: NvFp4ScaleMode::Multiplier,
        },
    )
}

fn upload(backend: &CudaBackend, catalog: &TensorCatalog) -> Result<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    for base in [GATE, UP, QUERY, KEY, VALUE] {
        let base =
            base.strip_suffix(".weight").ok_or(Error::InvalidNvFp4("invalid weight name"))?;
        for name in [
            format!("{base}.weight"),
            format!("{base}.weight_scale"),
            format!("{base}.weight_scale_2"),
            format!("{base}.input_scale"),
        ] {
            let info = catalog
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .ok_or_else(|| Error::MissingTensor(name.clone()))?;
            upload.enqueue(info)?;
        }
    }
    upload.finish()
}

fn get<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}

fn copy_device<T: DeviceElement + Copy>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read_device<T: DeviceElement + Copy>(
    backend: &CudaBackend,
    device: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(device.len())?;
    backend.inner.stream.copy_to_host(device, &mut host)?;
    Ok(host.to_vec()?)
}
