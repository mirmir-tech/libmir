//! The operations a decoder layer runs, with the parameters loaders need.
//!
//! The lowering is total over the semantic spec and fails only where the
//! spec names what no kernel computes; each backend adds its kernels' limits
//! on top.

mod feed_forward;

pub use feed_forward::{Activation, Clamp, Experts, FeedForwardLowering, lower_feed_forward};
use models::semantic::{AttentionWindow, KeyValueRelation, MixerSpec, QkNormalization};

use crate::capability::MixerKind;

/// The geometry of multi-head latent attention a kernel is chosen by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentGeometry {
    pub key_value_rank: usize,
    pub rope_head_dim: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixerLowering {
    Softmax {
        sinks: bool,
        window: Option<AttentionWindow>,
        normalization: QkNormalization,
        key_value_relation: KeyValueRelation,
    },
    Linear,
    StateSpace,
    Latent(LatentGeometry),
}

impl MixerLowering {
    #[must_use]
    pub const fn lower(spec: &MixerSpec) -> Self {
        match spec {
            MixerSpec::SoftmaxAttention(attention) => Self::Softmax {
                sinks: attention.sinks,
                window: attention.window,
                normalization: attention.qk_normalization,
                key_value_relation: attention.key_value_relation,
            },
            MixerSpec::LinearAttention(_) => Self::Linear,
            MixerSpec::StateSpace(_) => Self::StateSpace,
            MixerSpec::LatentAttention(latent) => Self::Latent(LatentGeometry {
                key_value_rank: latent.key_value_rank,
                rope_head_dim: latent.rope_head_dim,
            }),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> MixerKind {
        match *self {
            Self::Softmax { sinks, window, .. } => MixerKind::softmax(sinks, window),
            Self::Linear => MixerKind::Linear,
            Self::StateSpace => MixerKind::StateSpace,
            Self::Latent(_) => MixerKind::Latent,
        }
    }

    /// The Q/K normalization of a softmax mixer.
    #[must_use]
    pub const fn normalization(&self) -> Option<QkNormalization> {
        match *self {
            Self::Softmax { normalization, .. } => Some(normalization),
            Self::Linear | Self::StateSpace | Self::Latent(_) => None,
        }
    }
}
