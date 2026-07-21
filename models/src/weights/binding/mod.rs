mod dense;
mod dimensions;
mod discovery;
mod grammar;
mod hybrid;
mod roles;
mod shapes;
mod types;
mod validation;
mod view;

pub use dense::{DenseDecoderLayerBindings, DenseSoftmaxBindings};
pub use hybrid::{
    GatedSoftmaxBindings, HybridDecoderLayerBindings, HybridMixerBindings, LinearAttentionBindings,
    SharedRoutedFeedForwardBindings,
    moe::{
        HybridMoeAttentionBindings, HybridMoeDenseBindings, HybridMoeExpertBindings,
        HybridMoeLayerBindings, HybridMoeRouterBindings,
    },
};
pub use roles::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    LinearAttentionTensorRole, LogicalTensorRole,
};
pub use types::{
    BindingTransform, BlockFormat, ExpertProjectionLayout, TensorBinding, TensorPacking,
    TensorStorage, WeightBindingPlan,
};
pub use view::{DecoderBoundaryBindings, RoutedDecoderLayerBindings, RoutedExpertBindings};

use crate::{error::Result, semantic::SemanticModelSpec, weights::TensorCatalog};

impl WeightBindingPlan {
    pub fn discover(spec: &SemanticModelSpec, catalog: &TensorCatalog) -> Result<Self> {
        discovery::discover(spec, catalog)
    }

    #[must_use]
    pub fn binding(&self, role: &LogicalTensorRole) -> Option<&TensorBinding> {
        self.tensors.iter().find(|binding| &binding.role == role)
    }

    #[must_use]
    pub fn expert_projection_layout(&self) -> Option<ExpertProjectionLayout> {
        let interleaved = self.tensors.iter().any(|binding| {
            binding
                .transforms
                .contains(&BindingTransform::FusedGateUp { interleaved: true })
        });
        let gate = self
            .tensors
            .iter()
            .any(|binding| stacked_expert(binding, ExpertProjectionRole::Gate));
        let up = self
            .tensors
            .iter()
            .any(|binding| stacked_expert(binding, ExpertProjectionRole::Up));
        let separate = gate && up;
        match (interleaved, separate) {
            (true, false) => Some(ExpertProjectionLayout::InterleavedGateUp),
            (false, true) => Some(ExpertProjectionLayout::SeparateGateUp),
            (false, false) | (true, true) => None,
        }
    }

    #[must_use]
    pub fn uses_block_format(&self, format: BlockFormat) -> bool {
        self.tensors.iter().any(|binding| {
            matches!(binding.storage, TensorStorage::BlockQuantized { format: value, .. } if value == format)
        })
    }

    #[must_use]
    pub fn affine_group_size(&self) -> Option<usize> {
        dimensions::uniform_affine_group_size(&self.tensors)
    }
}

fn stacked_expert(binding: &TensorBinding, projection: ExpertProjectionRole) -> bool {
    matches!(
        binding.role,
        LogicalTensorRole::Layer {
            tensor: LayerTensorRole::ExpertProjection {
                expert: None,
                projection: value,
            },
            ..
        } if value == projection
    )
}

#[cfg(test)]
mod tests;
