//! CUDA decoder runtimes, chosen by what they execute rather than by the
//! order of an `if` chain.

mod table;

use super::CudaDecoderPlan;
use crate::{ArchitectureError, Rejection, Result, Runtime, capability::Operations};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CudaDecoderRuntime {
    Dense,
    DenseAndRouted,
    SharedRouted,
    DenseMixed,
    ClampedRouted,
    /// Mamba-2 and attention mixers with latent routed experts (Nemotron-H).
    StateSpaceHybrid,
    /// Multi-head latent attention with dense or shared-routed feed-forwards.
    LatentAttention,
}

impl CudaDecoderRuntime {
    pub const ALL: [Self; 7] = [
        Self::Dense,
        Self::DenseAndRouted,
        Self::SharedRouted,
        Self::DenseMixed,
        Self::ClampedRouted,
        Self::StateSpaceHybrid,
        Self::LatentAttention,
    ];

    /// Which runtime runs a plan several execute: the higher one. The graph
    /// decoder comes first, since the mixed and shared-routed runtimes also
    /// execute plain softmax decoders; every priority is distinct.
    #[must_use]
    pub const fn priority(self) -> u8 {
        match self {
            Self::Dense => 7,
            Self::LatentAttention => 6,
            Self::StateSpaceHybrid => 5,
            Self::ClampedRouted => 4,
            Self::DenseAndRouted => 3,
            Self::SharedRouted => 2,
            Self::DenseMixed => 1,
        }
    }

    /// Whether the runtime reallocates its K/V pages after loading, so a
    /// provisional cache and a measured resize are possible.
    #[must_use]
    pub const fn resizable_kv_cache(self) -> bool {
        match self {
            Self::SharedRouted
            | Self::DenseMixed
            | Self::StateSpaceHybrid
            | Self::LatentAttention => true,
            Self::Dense | Self::DenseAndRouted | Self::ClampedRouted => false,
        }
    }

    /// What the runtime executes.
    #[must_use]
    pub fn capabilities(self) -> Operations {
        table::capabilities(self)
    }

    /// The highest-priority runtime that executes `plan`, or every runtime's
    /// reasons for refusing it.
    pub fn select(plan: &CudaDecoderPlan) -> Result<Self> {
        let required = plan.requirements();
        let mut rejections = Vec::new();
        let mut selected: Option<Self> = None;
        for runtime in Self::ALL {
            let missing = runtime.capabilities().missing(&required);
            if !missing.is_empty() {
                rejections.push(Rejection { runtime: Runtime::Cuda(runtime), missing });
            } else if selected.is_none_or(|best| runtime.priority() > best.priority()) {
                selected = Some(runtime);
            }
        }
        selected.ok_or(ArchitectureError::NoRuntime { rejections })
    }
}
