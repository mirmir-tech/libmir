mod admission;
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
                    CudaFeedForwardLowering::Routed { shared: false, clamped: false },
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
pub enum CudaQkvNormalization {
    None,
    All,
    QueryKey,
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
    /// Multi-head latent attention with dense or shared-routed feed-forwards
    /// (`DeepSeek` V2/V3).
    LatentAttention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaArchitecture {
    Generation(CudaDecoderRuntime),
    Embedding,
    SequenceScoring,
}

/// The one Q/K/V normalization of every graph-decoder layer. Layers whose
/// keys double as values bind the key projection twice, so the K/V relation
/// may differ between layers.
pub fn graph_normalization(plan: &CudaDecoderPlan) -> Result<CudaQkvNormalization> {
    let normalizations = plan
        .layers
        .iter()
        .map(|layer| match layer.mixer {
            CudaMixerLowering::Softmax { normalization, .. } => Ok(normalization),
            CudaMixerLowering::Linear
            | CudaMixerLowering::StateSpace
            | CudaMixerLowering::Latent => Err(ArchitectureError::invalid(
                "CUDA graph decoder requires softmax attention in every layer",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(&first) = normalizations.first() else {
        return Err(ArchitectureError::invalid("CUDA graph decoder has no layers"));
    };
    if normalizations.iter().any(|normalization| *normalization != first) {
        return Err(ArchitectureError::invalid(
            "CUDA graph decoder requires uniform Q/K/V normalization",
        ));
    }
    Ok(match first {
        QkNormalization::None => CudaQkvNormalization::None,
        QkNormalization::QueryKeyRms => CudaQkvNormalization::QueryKey,
        QkNormalization::QueryKeyValueRms => CudaQkvNormalization::All,
    })
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
