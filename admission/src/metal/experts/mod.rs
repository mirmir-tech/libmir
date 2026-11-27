//! The routed-expert layouts Metal's loaders build expert banks from. The
//! loaders dispatch on these, and dense admission checks them, so a
//! checkpoint is admitted only in a layout its runtime's loader reads.

use models::weights::{
    BlockQuantization, ExpertProjectionRole, HybridMoeExpertBindings, RoutedExpertBindings,
    TensorBinding, TensorStorage,
};

use super::MetalDecoderRuntime;

/// How per-expert projections stack into one `[experts, output, input]`
/// bank: every expert of the bank in the same storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndividualBank {
    /// BF16/F16/F32 matrices, gathered by a dense product.
    Dense,
    /// NVFP4 blocks with their scales.
    NvFp4,
    /// Compressed-tensors packed integers, stacked for MLX's `gather_qmm`.
    PackedInteger,
}

impl IndividualBank {
    /// The bank `bindings` stack into; none when empty, mixed, or in a
    /// storage no bank is built from.
    #[must_use]
    pub fn of(bindings: &[&TensorBinding]) -> Option<Self> {
        let kind = |binding: &TensorBinding| match &binding.storage {
            TensorStorage::Dense { .. } => Some(Self::Dense),
            TensorStorage::BlockQuantized { format: BlockQuantization::NVFP4, .. } => {
                Some(Self::NvFp4)
            },
            TensorStorage::PackedInteger { .. } => Some(Self::PackedInteger),
            _ => None,
        };
        let first = kind(bindings.first()?)?;
        bindings.iter().all(|binding| kind(binding) == Some(first)).then_some(first)
    }
}

/// The loader a runtime reads its routed experts with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedExpertLoader {
    /// `SharedExpertMoe`: gated experts, separate, fused or per expert.
    Gated,
    /// `ClampedRoutedExperts` (GPT-OSS): gated experts stacked by expert.
    GatedStacked,
    /// The state-space hybrid's ungated up/down experts, stacked or per
    /// expert.
    Ungated,
}

impl RoutedExpertLoader {
    /// The loader `runtime` reads routed experts with; none for the hybrid
    /// runtimes, which read [`HybridMoeExpertBindings`] (see
    /// [`hybrid_experts_load`]).
    #[must_use]
    pub const fn of(runtime: MetalDecoderRuntime) -> Option<Self> {
        match runtime {
            MetalDecoderRuntime::Dense
            | MetalDecoderRuntime::SharedRouted
            | MetalDecoderRuntime::DenseMixed
            | MetalDecoderRuntime::LatentAttention => Some(Self::Gated),
            MetalDecoderRuntime::ClampedRouted => Some(Self::GatedStacked),
            MetalDecoderRuntime::StateSpaceHybrid => Some(Self::Ungated),
            MetalDecoderRuntime::DenseAndRouted | MetalDecoderRuntime::DenseGelu => None,
        }
    }

    /// Whether this loader reads `experts`.
    #[must_use]
    pub fn loads(self, experts: RoutedExpertBindings<'_>) -> bool {
        match (self, experts) {
            (
                Self::Gated | Self::GatedStacked,
                RoutedExpertBindings::InterleavedGateUp { .. }
                | RoutedExpertBindings::SeparateGateUp { .. },
            )
            | (Self::Ungated, RoutedExpertBindings::UngatedStacked { .. }) => true,
            (Self::Gated, RoutedExpertBindings::Individual { .. }) => {
                [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                    .into_iter()
                    .all(|role| IndividualBank::of(&experts.individual(role)).is_some())
            },
            (Self::Ungated, RoutedExpertBindings::Individual { .. }) => {
                [ExpertProjectionRole::Up, ExpertProjectionRole::Down]
                    .into_iter()
                    .all(|role| IndividualBank::of(&experts.individual(role)).is_some())
            },
            (Self::Gated | Self::GatedStacked, RoutedExpertBindings::UngatedStacked { .. })
            | (Self::GatedStacked, RoutedExpertBindings::Individual { .. })
            | (
                Self::Ungated,
                RoutedExpertBindings::InterleavedGateUp { .. }
                | RoutedExpertBindings::SeparateGateUp { .. },
            ) => false,
        }
    }
}

/// Whether `HybridMoeModel` reads `experts`: stacked, or per expert in a
/// storage a bank is built from.
#[must_use]
pub fn hybrid_experts_load(experts: &HybridMoeExpertBindings<'_>) -> bool {
    match experts {
        HybridMoeExpertBindings::Stacked(_) | HybridMoeExpertBindings::FusedStacked { .. } => true,
        HybridMoeExpertBindings::Individual { gate, up, down } => {
            [gate, up, down].into_iter().all(|bank| IndividualBank::of(bank).is_some())
        },
    }
}

#[cfg(test)]
mod tests;
