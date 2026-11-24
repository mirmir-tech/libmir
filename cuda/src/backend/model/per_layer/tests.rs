//! Per-layer inputs against a host reference of Hugging Face's
//! `project_per_layer_inputs`.

use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::{PerLayerInputTensors, PerLayerInputs, PerLayerInputsTemplate};
use crate::{
    CudaBackend, CudaConfig, CudaTensor, Result, backend::linear::CheckpointProjectionWeight,
};

const VOCAB: usize = 8;
const LAYERS: usize = 3;
const WIDTH: usize = 32;
const HIDDEN: usize = 64;
const EPSILON: f32 = 1.0e-6;

fn values(elements: usize, seed: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let step = u16::try_from((index * 7 + seed * 13) % 29).unwrap_or(0);
            bf16::from_f32(f32::from(step) / 16.0 - 0.875)
        })
        .collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}

fn read(backend: &CudaBackend, source: &DeviceBuffer<bf16>) -> Result<Vec<f32>> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?.into_iter().map(bf16::to_f32).collect())
}

fn tensor(backend: &CudaBackend, name: &str, shape: &[usize], seed: usize) -> Result<CudaTensor> {
    let elements = shape.iter().product();
    let buffer = copy(backend, &values(elements, seed))?;
    Ok(CudaTensor::from_bf16(name.into(), shape.to_vec(), buffer))
}

/// `inputs[l][t]` of the host reference, layer-major.
fn reference(tokens: &[u32], embeddings: &[bf16], tables: [&[bf16]; 3]) -> Vec<f32> {
    let [table, projection, norm] = tables;
    let total = LAYERS * WIDTH;
    let width = f32::from(u16::try_from(WIDTH).unwrap_or(1));
    let embedding_scale = width.sqrt();
    let projection_scale = f32::from(u16::try_from(HIDDEN).unwrap_or(1)).sqrt().recip();
    let mut output = vec![0.0; tokens.len() * total];
    for (row, token) in tokens.iter().enumerate() {
        let embedding = &embeddings[row * HIDDEN..(row + 1) * HIDDEN];
        let projected = (0..total)
            .map(|column| {
                let weights = &projection[column * HIDDEN..(column + 1) * HIDDEN];
                let sum = weights.iter().zip(embedding).map(|(w, x)| w.to_f32() * x.to_f32());
                bf16::from_f32(sum.sum::<f32>()).to_f32() * projection_scale
            })
            .collect::<Vec<_>>();
        for layer in 0..LAYERS {
            let slice = &projected[layer * WIDTH..(layer + 1) * WIDTH];
            let mean = slice.iter().map(|value| value * value).sum::<f32>() / width;
            let inverse = (mean + EPSILON).sqrt().recip();
            for column in 0..WIDTH {
                let index = usize::try_from(*token).unwrap_or(0) * total + layer * WIDTH + column;
                let embedded = bf16::from_f32(table[index].to_f32() * embedding_scale).to_f32();
                let normalized = slice[column] * inverse * norm[column].to_f32();
                output[(layer * tokens.len() + row) * WIDTH + column] =
                    (normalized + embedded) * std::f32::consts::FRAC_1_SQRT_2;
            }
        }
    }
    output
}

#[test]
fn computes_layer_major_inputs_as_the_reference() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let total = LAYERS * WIDTH;
    let table = tensor(&backend, "per_layer.embedding", &[VOCAB, total], 1)?;
    let projection = tensor(&backend, "per_layer.projection", &[total, HIDDEN], 2)?;
    let norm = tensor(&backend, "per_layer.norm", &[WIDTH], 3)?;
    let template = PerLayerInputsTemplate::new(
        PerLayerInputTensors {
            embedding: CheckpointProjectionWeight::Dense(table.clone()),
            model_projection: projection.clone(),
            projection_norm: norm.clone(),
        },
        [VOCAB, LAYERS, WIDTH, HIDDEN],
        EPSILON,
    )?;
    let tokens = [1_u32, 5, 7];
    let embeddings = values(tokens.len() * HIDDEN, 4);
    let mut inputs = PerLayerInputs::new(&backend, &template, 1)?;
    inputs.ensure_capacity(tokens.len())?;
    inputs.compute(&copy(&backend, &tokens)?, 0, tokens.len(), &copy(&backend, &embeddings)?)?;
    let host = |tensor: &CudaTensor| -> Result<Vec<bf16>> {
        let buffer = tensor.as_bf16().ok_or(crate::Error::InvalidDecoderKernel("not BF16"))?;
        Ok(read(&backend, buffer)?.into_iter().map(bf16::from_f32).collect())
    };
    let expected =
        reference(&tokens, &embeddings, [&host(&table)?, &host(&projection)?, &host(&norm)?]);
    for layer in 0..LAYERS {
        let actual = read(&backend, &inputs.layer(layer, tokens.len())?)?;
        let rows = layer * tokens.len() * WIDTH..(layer + 1) * tokens.len() * WIDTH;
        for (index, (actual, expected)) in actual.iter().zip(&expected[rows]).enumerate() {
            let tolerance = 1.0e-2 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance, "layer {layer} [{index}]");
        }
    }
    Ok(())
}
