mod admission;
mod dense;
mod latent;
pub use admission::admit;
use models::semantic::{
    ActivationSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, QkNormalization,
    SemanticModelSpec,
};

use crate::{ArchitectureError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaMixerLowering {
    Softmax {
        sinks: bool,
        normalization: QkNormalization,
        key_value_relation: KeyValueRelation,
    },
    Linear,
    StateSpace,
    /// Multi-head latent attention; no CUDA runtime executes it yet.
    Latent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaFeedForwardLowering {
    Dense,
    Routed { shared: bool, clamped: bool },
    DenseAndRouted,
    LatentRouted { shared: bool },
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaLayerLowering {
    pub mixer: CudaMixerLowering,
    pub feed_forward: CudaFeedForwardLowering,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDecoderPlan {
    layers: Vec<CudaLayerLowering>,
}

impl CudaDecoderPlan {
    #[must_use]
    pub fn lower(spec: &SemanticModelSpec) -> Self {
        let layers = spec
            .decoder
            .layers
            .iter()
            .map(|layer| CudaLayerLowering {
                mixer: lower_mixer(&layer.mixer),
                feed_forward: lower_feed_forward(&layer.feed_forward),
            })
            .collect();
        Self { layers }
    }

    #[must_use]
    pub fn layers(&self) -> &[CudaLayerLowering] {
        &self.layers
    }

    #[must_use]
    pub fn all_dense(&self) -> bool {
        self.all_feed_forward(CudaFeedForwardLowering::Dense)
    }

    /// Softmax attention in every layer with a gated MLP or, without sinks,
    /// routed experts that have no shared expert and no clamp: the two
    /// feed-forwards of the dense runtime.
    #[must_use]
    pub fn all_dense_or_routed(&self) -> bool {
        !self.layers.is_empty()
            && self.layers.iter().all(|layer| match (layer.mixer, layer.feed_forward) {
                (CudaMixerLowering::Softmax { .. }, CudaFeedForwardLowering::Dense) => true,
                (
                    CudaMixerLowering::Softmax { sinks, .. },
                    CudaFeedForwardLowering::Routed { clamped: false, .. },
                ) => !sinks,
                _ => false,
            })
    }

    #[must_use]
    pub fn all_dense_and_routed(&self) -> bool {
        self.all_feed_forward(CudaFeedForwardLowering::DenseAndRouted)
    }

    #[must_use]
    pub fn all_shared_routed(&self) -> bool {
        self.layers.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                CudaFeedForwardLowering::Routed { shared: true, clamped: false }
            )
        })
    }

    #[must_use]
    pub fn all_unshared_clamped_routed(&self) -> bool {
        self.layers.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                CudaFeedForwardLowering::Routed { shared: false, clamped: true }
            ) && matches!(layer.mixer, CudaMixerLowering::Softmax { sinks: true, .. })
        })
    }

    #[must_use]
    pub fn has_linear_mixer(&self) -> bool {
        self.layers.iter().any(|layer| layer.mixer == CudaMixerLowering::Linear)
    }

    #[must_use]
    pub fn has_latent_mixer(&self) -> bool {
        self.layers.iter().any(|layer| layer.mixer == CudaMixerLowering::Latent)
    }

    #[must_use]
    pub fn has_state_space_mixer(&self) -> bool {
        self.layers.iter().any(|layer| layer.mixer == CudaMixerLowering::StateSpace)
    }

    #[must_use]
    pub fn has_softmax_mixer(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer.mixer, CudaMixerLowering::Softmax { .. }))
    }

    fn all_feed_forward(&self, expected: CudaFeedForwardLowering) -> bool {
        !self.layers.is_empty()
            && self.layers.iter().all(|layer| layer.feed_forward == expected)
            && self
                .layers
                .iter()
                .all(|layer| matches!(layer.mixer, CudaMixerLowering::Softmax { .. }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaArchitecture {
    Generation(CudaDecoderRuntime),
    Embedding,
    SequenceScoring,
}

/// Softmax attention in every layer of the graph decoder.
///
/// Each layer keeps its own Q/K/V normalization (Llama 4 normalizes its
/// rotary layers only), and layers whose keys double as values bind the key
/// projection twice.
pub fn require_graph_attention(plan: &CudaDecoderPlan) -> Result<()> {
    if plan.layers.is_empty() {
        return Err(ArchitectureError::invalid("CUDA graph decoder has no layers"));
    }
    if plan
        .layers
        .iter()
        .all(|layer| matches!(layer.mixer, CudaMixerLowering::Softmax { .. }))
    {
        Ok(())
    } else {
        Err(ArchitectureError::invalid(
            "CUDA graph decoder requires softmax attention in every layer",
        ))
    }
}

fn lower_mixer(spec: &MixerSpec) -> CudaMixerLowering {
    match spec {
        MixerSpec::SoftmaxAttention(attention) => CudaMixerLowering::Softmax {
            sinks: attention.sinks,
            normalization: attention.qk_normalization,
            key_value_relation: attention.key_value_relation,
        },
        MixerSpec::LinearAttention(_) => CudaMixerLowering::Linear,
        MixerSpec::StateSpace(_) => CudaMixerLowering::StateSpace,
        MixerSpec::LatentAttention(_) => CudaMixerLowering::Latent,
    }
}

fn lower_feed_forward(spec: &FeedForwardSpec) -> CudaFeedForwardLowering {
    match spec {
        FeedForwardSpec::Dense { .. } => CudaFeedForwardLowering::Dense,
        FeedForwardSpec::DenseAndRouted { .. } => CudaFeedForwardLowering::DenseAndRouted,
        FeedForwardSpec::Routed { routed, shared } => CudaFeedForwardLowering::Routed {
            shared: shared.is_some(),
            clamped: matches!(
                routed.activation,
                ActivationSpec::SwiGlu { clamp: Some(_), up_shift, .. } if up_shift != 0.0
            ),
        },
        FeedForwardSpec::LatentRouted { shared, .. } => {
            CudaFeedForwardLowering::LatentRouted { shared: shared.is_some() }
        },
        FeedForwardSpec::None => CudaFeedForwardLowering::None,
    }
}
