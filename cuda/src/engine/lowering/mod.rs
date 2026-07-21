use models::semantic::{
    ActivationSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, QkNormalization,
    SemanticModelSpec,
};

use crate::{Error, Result, kernels::QkvNormalization as KernelNormalization};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerLowering {
    Softmax {
        sinks: bool,
        normalization: QkNormalization,
        key_value_relation: KeyValueRelation,
    },
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardLowering {
    Dense,
    Routed { shared: bool, clamped: bool },
    DenseAndRouted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerLowering {
    pub mixer: MixerLowering,
    pub feed_forward: FeedForwardLowering,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDecoderPlan {
    layers: Vec<LayerLowering>,
}

impl CudaDecoderPlan {
    #[must_use]
    pub fn lower(spec: &SemanticModelSpec) -> Self {
        let layers = spec
            .decoder
            .layers
            .iter()
            .map(|layer| LayerLowering {
                mixer: lower_mixer(&layer.mixer),
                feed_forward: lower_feed_forward(&layer.feed_forward),
            })
            .collect();
        Self { layers }
    }

    #[must_use]
    pub fn layers(&self) -> &[LayerLowering] {
        &self.layers
    }

    #[must_use]
    pub fn all_dense(&self) -> bool {
        self.all_feed_forward(FeedForwardLowering::Dense)
    }

    #[must_use]
    pub fn all_dense_and_routed(&self) -> bool {
        self.all_feed_forward(FeedForwardLowering::DenseAndRouted)
    }

    #[must_use]
    pub fn all_shared_routed(&self) -> bool {
        self.layers.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                FeedForwardLowering::Routed { shared: true, clamped: false }
            )
        })
    }

    #[must_use]
    pub fn all_unshared_clamped_routed(&self) -> bool {
        self.layers.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                FeedForwardLowering::Routed { shared: false, clamped: true }
            ) && matches!(layer.mixer, MixerLowering::Softmax { sinks: true, .. })
        })
    }

    #[must_use]
    pub fn has_linear_mixer(&self) -> bool {
        self.layers.iter().any(|layer| layer.mixer == MixerLowering::Linear)
    }

    #[must_use]
    pub fn has_softmax_mixer(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer.mixer, MixerLowering::Softmax { .. }))
    }

    pub fn graph_normalization(&self) -> Result<KernelNormalization> {
        let softmax = self
            .layers
            .iter()
            .map(|layer| match layer.mixer {
                MixerLowering::Softmax { normalization, key_value_relation, .. } => {
                    Ok((normalization, key_value_relation))
                },
                MixerLowering::Linear => Err(Error::MissingCapability {
                    operation: "graph decoder softmax attention",
                    storage: "CUDA graph decoder layer plan".into(),
                    geometry: format!("layers={}", self.layers.len()),
                    requirement: "every graph-decoder layer must use softmax attention",
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        if softmax.iter().all(|value| *value == (QkNormalization::None, value.1)) {
            return Ok(KernelNormalization::NONE);
        }
        if softmax
            .iter()
            .all(|value| *value == (QkNormalization::QueryKeyRms, KeyValueRelation::KeyEqualsValue))
        {
            return Ok(KernelNormalization::ALL);
        }
        if softmax
            .iter()
            .all(|value| *value == (QkNormalization::QueryKeyRms, KeyValueRelation::Separate))
        {
            return Ok(KernelNormalization::QUERY_KEY);
        }
        Err(Error::MissingCapability {
            operation: "graph decoder Q/K normalization",
            storage: "CUDA graph decoder layer plan".into(),
            geometry: format!("layers={}", self.layers.len()),
            requirement: "Q/K normalization and K/V relation must be uniform across layers",
        })
    }

    fn all_feed_forward(&self, expected: FeedForwardLowering) -> bool {
        !self.layers.is_empty()
            && self.layers.iter().all(|layer| layer.feed_forward == expected)
            && self
                .layers
                .iter()
                .all(|layer| matches!(layer.mixer, MixerLowering::Softmax { .. }))
    }
}

fn lower_mixer(spec: &MixerSpec) -> MixerLowering {
    match spec {
        MixerSpec::SoftmaxAttention(attention) => MixerLowering::Softmax {
            sinks: attention.sinks,
            normalization: attention.qk_normalization,
            key_value_relation: attention.key_value_relation,
        },
        MixerSpec::LinearAttention(_) => MixerLowering::Linear,
    }
}

fn lower_feed_forward(spec: &FeedForwardSpec) -> FeedForwardLowering {
    match spec {
        FeedForwardSpec::Dense { .. } => FeedForwardLowering::Dense,
        FeedForwardSpec::DenseAndRouted { .. } => FeedForwardLowering::DenseAndRouted,
        FeedForwardSpec::Routed { routed, shared } => FeedForwardLowering::Routed {
            shared: shared.is_some(),
            clamped: matches!(
                routed.activation,
                ActivationSpec::SwiGlu { clamp: Some(_), up_shift, .. } if up_shift != 0.0
            ),
        },
    }
}

#[cfg(test)]
mod tests;
