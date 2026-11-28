//! Per-layer input weights load in the storage their bindings declare.

use mircuda::{DeviceElement, bf16};
use models::{
    gguf::GgmlType,
    weights::{
        GgmlBlockQuantization, GgmlCompanion, LayerTensorRole, LogicalTensorRole,
        PerLayerInputBindings, TensorBinding, TensorStorage,
    },
};

use super::load;
use crate::{
    CudaBackend, CudaConfig, CudaTensor, CudaTensorSet, Error, Result,
    backend::CheckpointProjectionWeight, kernels::GgmlBlocks,
};

const HIDDEN: usize = 64;
const WIDTH: usize = 32;
/// Bytes of one `Q8_0` block of 32 values.
const BLOCK: usize = 34;

fn binding(source: &str, shape: &[usize], storage: TensorStorage) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::InputNorm,
        },
        source: source.into(),
        shape: shape.to_vec(),
        logical_shape: Some(shape.to_vec()),
        transforms: Vec::new(),
        storage,
    }
}

fn q8_0() -> TensorStorage {
    TensorStorage::Ggml {
        format: GgmlBlockQuantization { ggml_type: GgmlType::Q8_0 },
        companion: GgmlCompanion::Plain,
    }
}

fn bf16_storage() -> TensorStorage {
    TensorStorage::Dense {
        dtype: models::weights::DenseDType::Bf16,
        bias: None,
    }
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<mircuda::DeviceBuffer<T>> {
    let mut host = backend.context().allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate::<T>(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    backend.stream().synchronize()?;
    Ok(device)
}

/// `[rows, columns]` of `Q8_0` blocks in the padded allocation the upload
/// makes, or as plain bytes.
fn ggml(
    backend: &CudaBackend,
    name: &str,
    [rows, columns]: [usize; 2],
    padded: bool,
) -> Result<CudaTensor> {
    let bytes = rows * columns / 32 * BLOCK;
    let shape = vec![rows, columns / 32 * BLOCK];
    if !padded {
        return Ok(CudaTensor::from_u8(name.into(), shape, copy(backend, &vec![0_u8; bytes])?));
    }
    let mut blocks = GgmlBlocks::allocate(backend.pool(), backend.stream(), bytes)?;
    let mut host = backend.context().allocate_pinned::<u8>(bytes)?;
    host.copy_from_slice(&vec![0_u8; bytes])?;
    backend.stream().copy_to_device(&mut host, blocks.view_mut())?;
    backend.stream().synchronize()?;
    Ok(CudaTensor::from_ggml_blocks(name.into(), shape, blocks))
}

fn bf16_tensor(backend: &CudaBackend, name: &str, shape: &[usize]) -> Result<CudaTensor> {
    let values = vec![bf16::ONE; shape.iter().product()];
    Ok(CudaTensor::from_bf16(name.into(), shape.to_vec(), copy(backend, &values)?))
}

/// Loads a gate and projection of `storage` over `tensors`.
fn loaded(
    backend: &CudaBackend,
    tensors: Vec<CudaTensor>,
    storage: fn() -> TensorStorage,
) -> Result<crate::PerLayerInputWeights> {
    let set = CudaTensorSet::from_tensors(backend.context(), backend.stream(), tensors);
    let gate = binding("inp_gate", &[WIDTH, HIDDEN], storage());
    let projection = binding("proj", &[HIDDEN, WIDTH], storage());
    let norm = binding("post_norm", &[HIDDEN], bf16_storage());
    load(
        &set,
        PerLayerInputBindings {
            gate: &gate,
            projection: &projection,
            output_norm: &norm,
        },
    )
}

/// Quantized GGUF `inp_gate` and `proj` load as GGML projections.
#[test]
fn loads_quantized_projections_as_ggml() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = vec![
        ggml(&backend, "inp_gate", [WIDTH, HIDDEN], true)?,
        ggml(&backend, "proj", [HIDDEN, WIDTH], true)?,
        bf16_tensor(&backend, "post_norm", &[HIDDEN])?,
    ];
    let weights = loaded(&backend, tensors, q8_0)?;
    assert!(matches!(weights.gate, CheckpointProjectionWeight::Ggml(_)));
    assert!(matches!(weights.projection, CheckpointProjectionWeight::Ggml(_)));
    Ok(())
}

/// Dense ones (F32 in GGUF files, cast to BF16 at upload) stay dense.
#[test]
fn loads_dense_projections_as_dense() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = vec![
        bf16_tensor(&backend, "inp_gate", &[WIDTH, HIDDEN])?,
        bf16_tensor(&backend, "proj", &[HIDDEN, WIDTH])?,
        bf16_tensor(&backend, "post_norm", &[HIDDEN])?,
    ];
    let weights = loaded(&backend, tensors, bf16_storage)?;
    assert!(matches!(weights.gate, CheckpointProjectionWeight::Dense(_)));
    assert!(matches!(weights.projection, CheckpointProjectionWeight::Dense(_)));
    Ok(())
}

/// Quantized bytes outside the padded block allocation are refused at load.
#[test]
fn refuses_unpadded_ggml_bytes() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let tensors = vec![
        ggml(&backend, "inp_gate", [WIDTH, HIDDEN], false)?,
        ggml(&backend, "proj", [HIDDEN, WIDTH], true)?,
        bf16_tensor(&backend, "post_norm", &[HIDDEN])?,
    ];
    let refused = loaded(&backend, tensors, q8_0);
    assert!(matches!(refused, Err(Error::DTypeMismatch { .. })));
    Ok(())
}
