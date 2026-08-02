use models::{
    execution::TaskExecutionPlan,
    layout::{EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind},
    semantic::{
        ActivationSpec, FeedForwardSpec, KeyValueRelation, MixerSpec, QkNormalization,
        SemanticModelSpec,
    },
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaFeedForwardLowering {
    Dense,
    Routed { shared: bool, clamped: bool },
    DenseAndRouted,
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
    ClampedRouted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaArchitecture {
    Generation(CudaDecoderRuntime),
    Embedding,
    SequenceScoring,
}

pub fn admit(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<CudaArchitecture> {
    match task {
        TaskExecutionPlan::Generation { .. } => generation(semantic),
        TaskExecutionPlan::Embedding { .. } => Ok(CudaArchitecture::Embedding),
        TaskExecutionPlan::SequenceScoring { encoder, .. } => sequence_scoring(encoder),
    }
}

fn sequence_scoring(config: &EncoderConfig) -> Result<CudaArchitecture> {
    let fixed_ntk =
        matches!(config.rope_scaling, Some(EncoderRopeScaling::Ntk { mixed_b: None, .. }));
    if config.packed_qkv
        && config.norm == NormKind::LayerNorm
        && config.hidden_activation == "gelu"
        && config.position_embedding == EncoderPositionEmbedding::Rope
        && config.type_vocab_size > 0
        && config.num_labels == 1
        && fixed_ntk
    {
        Ok(CudaArchitecture::SequenceScoring)
    } else {
        Err(ArchitectureError::invalid(
            "CUDA sequence scoring requires packed QKV, token types, LayerNorm, GELU, one label, and fixed NTK RoPE",
        ))
    }
}

pub fn graph_normalization(plan: &CudaDecoderPlan) -> Result<CudaQkvNormalization> {
    let softmax = plan
        .layers
        .iter()
        .map(|layer| match layer.mixer {
            CudaMixerLowering::Softmax { normalization, key_value_relation, .. } => {
                Ok((normalization, key_value_relation))
            },
            CudaMixerLowering::Linear => Err(ArchitectureError::invalid(
                "CUDA graph decoder requires softmax attention in every layer",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    if softmax.iter().all(|value| *value == (QkNormalization::None, value.1)) {
        return Ok(CudaQkvNormalization::None);
    }
    if softmax
        .iter()
        .all(|value| *value == (QkNormalization::QueryKeyRms, KeyValueRelation::KeyEqualsValue))
    {
        return Ok(CudaQkvNormalization::All);
    }
    if softmax
        .iter()
        .all(|value| *value == (QkNormalization::QueryKeyRms, KeyValueRelation::Separate))
    {
        return Ok(CudaQkvNormalization::QueryKey);
    }
    Err(ArchitectureError::invalid(
        "CUDA graph decoder requires uniform Q/K normalization and K/V relation",
    ))
}

fn generation(semantic: Option<&SemanticModelSpec>) -> Result<CudaArchitecture> {
    let semantic = semantic.ok_or_else(|| {
        ArchitectureError::invalid("CUDA generation semantic contract is missing")
    })?;
    let plan = CudaDecoderPlan::lower(semantic);
    let runtime = if plan.all_shared_routed() && plan.has_linear_mixer() && plan.has_softmax_mixer()
    {
        CudaDecoderRuntime::SharedRouted
    } else if plan.all_unshared_clamped_routed() {
        CudaDecoderRuntime::ClampedRouted
    } else if plan.all_dense_and_routed() {
        CudaDecoderRuntime::DenseAndRouted
    } else if plan.all_dense() {
        let _normalization = graph_normalization(&plan)?;
        CudaDecoderRuntime::Dense
    } else {
        return Err(ArchitectureError::invalid(format!(
            "CUDA has no runtime for the {}-layer decoder composition",
            plan.layers().len()
        )));
    };
    Ok(CudaArchitecture::Generation(runtime))
}

fn lower_mixer(spec: &MixerSpec) -> CudaMixerLowering {
    match spec {
        MixerSpec::SoftmaxAttention(attention) => CudaMixerLowering::Softmax {
            sinks: attention.sinks,
            normalization: attention.qk_normalization,
            key_value_relation: attention.key_value_relation,
        },
        MixerSpec::LinearAttention(_) => CudaMixerLowering::Linear,
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
    }
}
