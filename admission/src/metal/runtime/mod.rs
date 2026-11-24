//! Metal decoder runtimes, chosen by what they execute.

mod table;

use crate::{ArchitectureError, Rejection, Result, Runtime, capability::Operations};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MetalDecoderRuntime {
    Dense,
    /// Gemma 4-style gated GELU with value norms, output norms and a layer
    /// scale.
    DenseGelu,
    DenseAndRouted,
    SharedRouted,
    /// Dense `SwiGLU` layers mixing linear and softmax attention.
    DenseMixed,
    ClampedRouted,
    /// Mamba-2 and attention mixers with sigmoid-routed experts (Nemotron-H).
    StateSpaceHybrid,
    /// Multi-head latent attention with leading dense layers, then routed
    /// plus shared experts (`DeepSeek-V2/V3`, Kimi, Moonlight).
    LatentAttention,
}

impl MetalDecoderRuntime {
    pub const ALL: [Self; 8] = [
        Self::Dense,
        Self::DenseGelu,
        Self::DenseAndRouted,
        Self::SharedRouted,
        Self::DenseMixed,
        Self::ClampedRouted,
        Self::StateSpaceHybrid,
        Self::LatentAttention,
    ];

    /// Which runtime runs a plan several execute: the higher one. The dense
    /// runtime comes first, since the mixed and shared-routed runtimes also
    /// execute plain softmax decoders; every priority is distinct.
    #[must_use]
    pub const fn priority(self) -> u8 {
        match self {
            Self::Dense => 8,
            Self::LatentAttention => 7,
            Self::DenseGelu => 6,
            Self::DenseAndRouted => 5,
            Self::SharedRouted => 4,
            Self::DenseMixed => 3,
            Self::ClampedRouted => 2,
            Self::StateSpaceHybrid => 1,
        }
    }

    /// What the runtime executes.
    #[must_use]
    pub fn capabilities(self) -> Operations {
        table::capabilities(self)
    }

    /// The highest-priority runtime that executes `required`, or every
    /// runtime's reasons for refusing it.
    pub fn select(required: &Operations) -> Result<Self> {
        let mut rejections = Vec::new();
        let mut selected: Option<Self> = None;
        for runtime in Self::ALL {
            let missing = runtime.capabilities().missing(required);
            if !missing.is_empty() {
                rejections.push(Rejection {
                    runtime: Runtime::Metal(runtime),
                    missing,
                });
            } else if selected.is_none_or(|best| runtime.priority() > best.priority()) {
                selected = Some(runtime);
            }
        }
        selected.ok_or(ArchitectureError::NoRuntime { rejections })
    }
}
