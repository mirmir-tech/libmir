use mircuda::{DeviceBuffer, bf16};
use models::weights::{
    LatentAttentionBindings, LatentQueryBindings, LatentUpBindings, TensorBinding, TensorStorage,
};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, GgmlCheckpointWeight, Result,
    backend::linear::CheckpointProjectionWeight, kernels::LatentUpWeights,
};

/// Projections and norms of one latent attention layer.
#[derive(Clone, Debug)]
pub(super) struct LatentWeights {
    pub(super) input_norm: CudaTensor,
    pub(super) query: QueryWeights,
    pub(super) key_value_down: CheckpointProjectionWeight,
    pub(super) key_value_norm: CudaTensor,
    /// BF16 up-projections in their checkpoint layout, read in place by the
    /// absorption GEMMs.
    pub(super) up: LatentUp,
    pub(super) output: CheckpointProjectionWeight,
}

#[derive(Clone, Debug)]
pub(super) enum LatentUp {
    Fused(DeviceBuffer<bf16>),
    Split {
        key: DeviceBuffer<bf16>,
        value: DeviceBuffer<bf16>,
    },
}

impl LatentUp {
    pub(super) const fn weights(&self) -> LatentUpWeights<'_> {
        match self {
            Self::Fused(weights) => LatentUpWeights::Fused(weights),
            Self::Split { key, value } => LatentUpWeights::Split { key, value },
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum QueryWeights {
    Direct(Box<CheckpointProjectionWeight>),
    Latent(Box<LatentQueryWeights>),
}

/// Down-projection, normalized query latent, up-projection.
#[derive(Clone, Debug)]
pub(super) struct LatentQueryWeights {
    pub(super) down: CheckpointProjectionWeight,
    pub(super) norm: CudaTensor,
    pub(super) up: CheckpointProjectionWeight,
}

impl LatentWeights {
    pub(super) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        input_norm: &TensorBinding,
        bindings: LatentAttentionBindings<'_>,
    ) -> Result<Self> {
        let projection =
            |binding| CheckpointProjectionWeight::load_binding_prepared(backend, tensors, binding);
        let query = match bindings.query {
            LatentQueryBindings::Direct(query) => {
                QueryWeights::Direct(Box::new(projection(query)?))
            },
            LatentQueryBindings::Latent { down, norm, up } => {
                QueryWeights::Latent(Box::new(LatentQueryWeights {
                    down: projection(down)?,
                    norm: bf16(tensors, norm)?,
                    up: projection(up)?,
                }))
            },
        };
        Ok(Self {
            input_norm: bf16(tensors, input_norm)?,
            query,
            key_value_down: projection(bindings.key_value_down)?,
            key_value_norm: bf16(tensors, bindings.key_value_norm)?,
            up: match bindings.key_value_up {
                LatentUpBindings::Fused(fused) => LatentUp::Fused(up(backend, tensors, fused)?),
                LatentUpBindings::Split { key, value } => LatentUp::Split {
                    key: up(backend, tensors, key)?,
                    value: up(backend, tensors, value)?,
                },
            },
            output: projection(bindings.output)?,
        })
    }
}

/// An up-projection in BF16: as stored, or dequantized once from GGUF.
fn up(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
) -> Result<DeviceBuffer<bf16>> {
    if matches!(binding.storage, TensorStorage::Ggml { .. }) {
        return GgmlCheckpointWeight::load_binding(tensors, binding)?.dequantize_bf16(backend);
    }
    bf16(tensors, binding)?.as_bf16().cloned().ok_or_else(|| Error::DTypeMismatch {
        name: binding.source.clone(),
        expected: "BF16",
    })
}

/// A BF16 tensor the kernels read directly.
fn bf16(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<CudaTensor> {
    let tensor = tensors
        .get(&binding.source)
        .cloned()
        .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch {
            name: binding.source.clone(),
            expected: "BF16",
        });
    }
    Ok(tensor)
}
