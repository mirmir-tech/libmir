use models::weights::{LatentAttentionBindings, LatentQueryBindings, TensorBinding};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    backend::linear::CheckpointProjectionWeight,
};

/// Projections and norms of one latent attention layer.
#[derive(Clone, Debug)]
pub(super) struct LatentWeights {
    pub(super) input_norm: CudaTensor,
    pub(super) query: QueryWeights,
    pub(super) key_value_down: CheckpointProjectionWeight,
    pub(super) key_value_norm: CudaTensor,
    /// BF16 `[heads · (nope + value), latent]`, read in place by the
    /// absorption GEMMs.
    pub(super) key_value_up: CudaTensor,
    pub(super) output: CheckpointProjectionWeight,
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
            key_value_up: bf16(tensors, bindings.key_value_up)?,
            output: projection(bindings.output)?,
        })
    }
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
