//! Feed-forward blocks with their resolved activations and routers.

use models::semantic::{
    ActivationSpec, ExpertGroupLimit, FeedForwardSpec, RoutedExpertsSpec, RouterNormalization,
};

use crate::{
    ArchitectureError, Result, Unlowerable,
    capability::{ActivationKind, ActivationSite, ActivationUse, FeedForwardShape, RouterKind},
};

/// gpt-oss's clamp of its `SwiGLU`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clamp {
    pub alpha: f64,
    pub limit: f64,
}

/// An activation a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    SwiGlu { clamp: Option<Clamp> },
    GeluTanh,
    ReluSquared,
}

impl Activation {
    /// Exhaustive: an activation kept only by name has no kernel.
    pub fn lower(spec: &ActivationSpec, layer: usize) -> Result<Self> {
        match spec {
            ActivationSpec::SwiGlu => Ok(Self::SwiGlu { clamp: None }),
            &ActivationSpec::ClampedSwiGlu { alpha, limit } => {
                Ok(Self::SwiGlu { clamp: Some(Clamp { alpha, limit }) })
            },
            ActivationSpec::GeluTanh => Ok(Self::GeluTanh),
            ActivationSpec::ReluSquared => Ok(Self::ReluSquared),
            ActivationSpec::NamedGated { name } => Err(ArchitectureError::Unlowerable {
                layer: Some(layer),
                feature: Unlowerable::Activation { name: name.clone() },
            }),
        }
    }

    #[must_use]
    pub const fn kind(self) -> ActivationKind {
        match self {
            Self::SwiGlu { clamp: None } => ActivationKind::SwiGlu,
            Self::SwiGlu { clamp: Some(_) } => ActivationKind::ClampedSwiGlu,
            Self::GeluTanh => ActivationKind::GeluTanh,
            Self::ReluSquared => ActivationKind::ReluSquared,
        }
    }
}

/// Routed experts as their kernels run them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experts {
    pub count: usize,
    pub top_k: usize,
    pub activation: Activation,
    pub router: RouterNormalization,
    pub group_limit: Option<ExpertGroupLimit>,
    pub output_scale: f64,
}

impl Experts {
    fn lower(spec: &RoutedExpertsSpec, layer: usize) -> Result<Self> {
        Ok(Self {
            count: spec.expert_count,
            top_k: spec.top_k,
            activation: Activation::lower(&spec.activation, layer)?,
            router: spec.router_normalization,
            group_limit: spec.group_limit,
            output_scale: spec.output_scale,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeedForwardLowering {
    Dense {
        activation: Activation,
    },
    Routed {
        experts: Experts,
        shared: Option<Activation>,
    },
    DenseAndRouted {
        dense: Activation,
        experts: Experts,
    },
    /// Routed experts in a projected latent space when `latent_size` is set,
    /// at the hidden width otherwise.
    LatentRouted {
        latent_size: Option<usize>,
        experts: Experts,
        shared: Option<Activation>,
    },
    None,
}

pub fn lower_feed_forward(spec: &FeedForwardSpec, layer: usize) -> Result<FeedForwardLowering> {
    let shared = |shared: &Option<models::semantic::SharedExpertSpec>| {
        shared
            .as_ref()
            .map(|shared| Activation::lower(&shared.activation, layer))
            .transpose()
    };
    Ok(match spec {
        FeedForwardSpec::Dense { activation, .. } => FeedForwardLowering::Dense {
            activation: Activation::lower(activation, layer)?,
        },
        FeedForwardSpec::Routed { routed, shared: expert } => FeedForwardLowering::Routed {
            experts: Experts::lower(routed, layer)?,
            shared: shared(expert)?,
        },
        FeedForwardSpec::DenseAndRouted { dense_activation, routed, .. } => {
            FeedForwardLowering::DenseAndRouted {
                dense: Activation::lower(dense_activation, layer)?,
                experts: Experts::lower(routed, layer)?,
            }
        },
        FeedForwardSpec::LatentRouted { latent_size, routed, shared: expert } => {
            FeedForwardLowering::LatentRouted {
                latent_size: *latent_size,
                experts: Experts::lower(routed, layer)?,
                shared: shared(expert)?,
            }
        },
        FeedForwardSpec::None => FeedForwardLowering::None,
    })
}

impl FeedForwardLowering {
    #[must_use]
    pub const fn shape(&self) -> FeedForwardShape {
        match self {
            Self::Dense { .. } => FeedForwardShape::Dense,
            Self::Routed { shared: None, .. } => FeedForwardShape::Routed,
            Self::Routed { shared: Some(_), .. } => FeedForwardShape::SharedRouted,
            Self::DenseAndRouted { .. } => FeedForwardShape::DenseAndRouted,
            Self::LatentRouted { shared: None, .. } => FeedForwardShape::LatentRouted,
            Self::LatentRouted { shared: Some(_), .. } => FeedForwardShape::SharedLatentRouted,
            Self::None => FeedForwardShape::None,
        }
    }

    /// The routed experts, when the block has any.
    #[must_use]
    pub const fn experts(&self) -> Option<&Experts> {
        match self {
            Self::Routed { experts, .. }
            | Self::DenseAndRouted { experts, .. }
            | Self::LatentRouted { experts, .. } => Some(experts),
            Self::Dense { .. } | Self::None => None,
        }
    }

    /// The router of the routed experts, when the block has any.
    #[must_use]
    pub fn router(&self) -> Option<RouterKind> {
        self.experts().map(|experts| RouterKind::of(&experts.router))
    }

    /// Every activation the block runs, at its site.
    pub fn activations(&self) -> impl Iterator<Item = ActivationUse> {
        let at = |site, activation: Activation| ActivationUse { site, kind: activation.kind() };
        let (dense, expert, shared) = match self {
            Self::Dense { activation } => (Some(*activation), None, None),
            Self::Routed { experts, shared } | Self::LatentRouted { experts, shared, .. } => {
                (None, Some(experts.activation), *shared)
            },
            Self::DenseAndRouted { dense, experts } => {
                (Some(*dense), Some(experts.activation), None)
            },
            Self::None => (None, None, None),
        };
        let dense = dense.map(|activation| at(ActivationSite::Dense, activation));
        let expert = expert.map(|activation| at(ActivationSite::Expert, activation));
        let shared = shared.map(|activation| at(ActivationSite::Shared, activation));
        dense.into_iter().chain(expert).chain(shared)
    }
}
