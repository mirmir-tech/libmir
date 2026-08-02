mod affine;
mod awq;
mod bitsandbytes;
mod block;
mod dense;
mod dimensions;
mod discovery;
mod float8;
mod gptq;
mod grammar;
mod hybrid;
mod packed_integer;
mod roles;
mod shapes;
mod types;
mod validation;
mod view;

pub use affine::{
    AffineBits, AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness,
    AffineStorageDType, AffineZeroPointMode, GroupedAffineQuantization,
};
pub use awq::{AwqBits, AwqPacking, AwqQuantization, AwqScaleDType, AwqStorageDType};
pub use bitsandbytes::{
    BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
    BitsAndBytesStorageDType,
};
pub use block::{
    BlockActivationMode, BlockFormat, BlockInputPadding, BlockProjectionLayout, BlockQuantization,
    BlockScale, BlockScaleEncoding, BlockStorageDType,
};
pub use dense::{DenseDecoderLayerBindings, DenseSoftmaxBindings};
pub use float8::{
    Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
    Float8ScaleGranularity, Float8ScaleMode,
};
pub use gptq::{
    GptqBits, GptqCheckpointFormat, GptqPacking, GptqQuantization, GptqScaleDType, GptqStorageDType,
};
pub use hybrid::{
    GatedSoftmaxBindings, HybridDecoderLayerBindings, HybridMixerBindings, LinearAttentionBindings,
    SharedRoutedFeedForwardBindings,
    moe::{
        HybridMoeAttentionBindings, HybridMoeDenseBindings, HybridMoeExpertBindings,
        HybridMoeLayerBindings, HybridMoeRouterBindings,
    },
};
pub use packed_integer::{
    CompressedIntegerActivationOrder, CompressedIntegerBits, CompressedIntegerPacking,
    CompressedIntegerQuantization, CompressedIntegerScaleDType, CompressedIntegerScaleStrategy,
    CompressedIntegerSignedness, CompressedIntegerStorageDType, CompressedIntegerZeroPointMode,
};
pub use roles::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LayerTensorRole,
    LinearAttentionTensorRole, LogicalTensorRole,
};
pub use types::{
    BindingTransform, ExpertProjectionLayout, TensorBinding, TensorPacking, TensorStorage,
    WeightBindingPlan,
};
pub use view::{DecoderBoundaryBindings, RoutedDecoderLayerBindings, RoutedExpertBindings};

use crate::{
    error::Result, layout::ModelLayout, semantic::SemanticModelSpec, weights::TensorCatalog,
};

impl WeightBindingPlan {
    pub fn discover(spec: &SemanticModelSpec, catalog: &TensorCatalog) -> Result<Self> {
        discovery::discover(spec, catalog)
    }

    pub fn discover_from_layout(
        spec: &SemanticModelSpec,
        catalog: &TensorCatalog,
        layout: &ModelLayout,
    ) -> Result<Self> {
        discovery::discover_from_layout(spec, catalog, layout)
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
            matches!(binding.storage, TensorStorage::BlockQuantized { format: value, .. } if value.format == format)
        })
    }

    #[must_use]
    pub fn uses_packed_int8(&self) -> bool {
        self.tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::PackedInt8 { .. }))
    }

    #[must_use]
    pub fn uses_packed_int4(&self) -> bool {
        self.tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::PackedInt4 { .. }))
    }

    #[must_use]
    pub fn uses_awq(&self) -> bool {
        self.tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::Awq { .. }))
    }

    #[must_use]
    pub fn uses_gptq(&self) -> bool {
        self.tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::Gptq { .. }))
    }

    #[must_use]
    pub fn uses_bitsandbytes_4bit(&self) -> bool {
        self.tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::BitsAndBytes4Bit { .. }))
    }

    #[must_use]
    pub fn uses_float8(&self) -> bool {
        self.tensors
            .iter()
            .any(|binding| matches!(binding.storage, TensorStorage::Float8 { .. }))
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
