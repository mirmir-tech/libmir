//! Gemma 4's per-layer input gate and projection (`blk.{i}.inp_gate` and
//! `blk.{i}.proj` in GGUF files), in whichever storage the checkpoint keeps
//! them: GGUF files may quantize them like the other projections.

use models::weights::{BlockFormat, PerLayerInputBindings, TensorBinding, TensorStorage};

use super::tensor;
use crate::{
    CudaTensorSet, Error, GgmlCheckpointWeight, PerLayerInputWeights, Result,
    backend::CheckpointProjectionWeight,
};

#[cfg(all(test, target_os = "linux"))]
mod tests;

/// The layer's per-layer input weights; the layer template checks their
/// dtypes and geometry against its config.
pub(super) fn load(
    tensors: &CudaTensorSet,
    bindings: PerLayerInputBindings<'_>,
) -> Result<PerLayerInputWeights> {
    Ok(PerLayerInputWeights {
        gate: projection(tensors, bindings.gate)?,
        projection: projection(tensors, bindings.projection)?,
        output_norm: tensor(tensors, &bindings.output_norm.source)?.clone(),
    })
}

fn projection(
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
) -> Result<CheckpointProjectionWeight> {
    match &binding.storage {
        // GGML projections add a companion output bias themselves.
        TensorStorage::Ggml { .. } => {
            GgmlCheckpointWeight::load_binding_with_bias(tensors, binding)
                .map(CheckpointProjectionWeight::Ggml)
        },
        // Layers share the gate's plan and bind its weights at execution,
        // which NVFP4's prepared weights cannot.
        TensorStorage::BlockQuantized { format, .. } if format.format == BlockFormat::NvFp4 => {
            Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA per-layer input projections cannot bind NVFP4 weights at execution: {}",
                binding.source
            )))
        },
        _ => CheckpointProjectionWeight::load_binding(tensors, binding),
    }
}
