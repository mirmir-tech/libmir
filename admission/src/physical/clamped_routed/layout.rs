//! The single layout CUDA's clamped-routed loader selects for the whole model.

use models::weights::{
    ExpertProjectionLayout, LayerTensorRole, LogicalTensorRole, TensorStorage, WeightBindingPlan,
};

/// The layout CUDA's clamped-routed loader selects for the whole model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaClampedExpertLayout {
    Ggml,
    Native,
    Mlx,
    Dense,
}

impl CudaClampedExpertLayout {
    /// Discovers the single CUDA clamped-routed expert layout from `bindings`.
    #[must_use]
    pub fn discover(bindings: &WeightBindingPlan) -> Option<Self> {
        if bindings.tensors.iter().any(|binding| {
            matches!(
                binding.role,
                LogicalTensorRole::Layer {
                    tensor: LayerTensorRole::ExpertProjection { .. },
                    ..
                }
            ) && binding.output_scale().is_none()
                && matches!(binding.storage, TensorStorage::Ggml { .. })
        }) {
            return Some(Self::Ggml);
        }
        if bindings.tensors.iter().any(|binding| {
            matches!(
                binding.role,
                LogicalTensorRole::Layer {
                    tensor: LayerTensorRole::ExpertProjection { .. },
                    ..
                }
            ) && matches!(binding.storage, TensorStorage::Dense { .. })
        }) {
            return Some(Self::Dense);
        }
        match bindings.expert_projection_layout() {
            Some(ExpertProjectionLayout::InterleavedGateUp) => Some(Self::Native),
            Some(ExpertProjectionLayout::SeparateGateUp) => Some(Self::Mlx),
            None => None,
        }
    }

    #[must_use]
    pub const fn storage(self) -> &'static str {
        match self {
            Self::Native => "interleaved MXFP4 gate/up blocks",
            Self::Mlx => "separate MLX affine MXFP4 gate/up blocks",
            Self::Dense => "dense BF16 selected-expert matrices",
            Self::Ggml => "GGML MXFP4 expert banks repacked to interleaved OCP blocks",
        }
    }
}
