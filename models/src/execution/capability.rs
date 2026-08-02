use std::collections::BTreeSet;

use super::{PoolingMode, TaskExecutionPlan};
use crate::semantic::{
    ActivationSpec, AttentionOutputSpec, FeedForwardSpec, MixerSpec, NormalizationKind,
    PositionEncodingSpec, QkNormalization, RopeScalingSpec, RotaryLayoutSpec, RouterNormalization,
    SemanticModelSpec,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ArchitectureCapability {
    GenerationTask,
    EmbeddingTask,
    SequenceScoringTask,
    ClsPooling,
    LastTokenPooling,
    MeanPooling,
    L2NormalizedEmbedding,
    PromptedEmbedding,
    RmsNormalization,
    LayerNormalization,
    SoftmaxAttention,
    LinearAttention,
    SlidingWindowAttention,
    AttentionSinks,
    QueryKeyRmsNormalization,
    SharedKeyValueProjection,
    GatedAttentionOutput,
    RotaryPositionEncoding,
    MultiSectionRotary,
    InterleavedMultiSectionRotary,
    PiecewiseRopeScaling,
    YarnRopeScaling,
    DenseFeedForward,
    RoutedExperts,
    SharedExpert,
    DenseAndRouted,
    SwiGlu,
    ClampedSwiGlu,
    GeluTanh,
    NamedGatedActivation,
    SoftmaxTopKRouter,
    UnitTopKRouter,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArchitectureRequirements {
    pub capabilities: Vec<ArchitectureCapability>,
}

impl ArchitectureRequirements {
    #[must_use]
    pub fn discover(task: &TaskExecutionPlan, semantic: Option<&SemanticModelSpec>) -> Self {
        let mut capabilities = BTreeSet::new();
        task_capabilities(task, &mut capabilities);
        if let Some(semantic) = semantic {
            decoder_capabilities(semantic, &mut capabilities);
        }
        Self {
            capabilities: capabilities.into_iter().collect(),
        }
    }
}

fn task_capabilities(
    task: &TaskExecutionPlan,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    match task {
        TaskExecutionPlan::Generation { .. } => {
            capabilities.insert(ArchitectureCapability::GenerationTask);
        },
        TaskExecutionPlan::Embedding { task, .. } => {
            capabilities.insert(ArchitectureCapability::EmbeddingTask);
            capabilities.insert(pooling(task.pooling));
            if task.normalize {
                capabilities.insert(ArchitectureCapability::L2NormalizedEmbedding);
            }
            if task.include_prompt || !task.prompts.is_empty() || task.default_prompt.is_some() {
                capabilities.insert(ArchitectureCapability::PromptedEmbedding);
            }
        },
        TaskExecutionPlan::SequenceScoring { task, .. } => {
            capabilities.insert(ArchitectureCapability::SequenceScoringTask);
            capabilities.insert(pooling(task.pooling));
        },
    }
}

const fn pooling(pooling: PoolingMode) -> ArchitectureCapability {
    match pooling {
        PoolingMode::Cls => ArchitectureCapability::ClsPooling,
        PoolingMode::LastToken => ArchitectureCapability::LastTokenPooling,
        PoolingMode::Mean => ArchitectureCapability::MeanPooling,
    }
}

fn decoder_capabilities(
    semantic: &SemanticModelSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    normalization(semantic.decoder.final_norm.kind, capabilities);
    for layer in &semantic.decoder.layers {
        normalization(layer.input_norm.kind, capabilities);
        normalization(layer.post_attention_norm.kind, capabilities);
        mixer(&layer.mixer, capabilities);
        feed_forward(&layer.feed_forward, capabilities);
    }
}

fn normalization(
    normalization: NormalizationKind,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    capabilities.insert(match normalization {
        NormalizationKind::Rms => ArchitectureCapability::RmsNormalization,
        NormalizationKind::Layer => ArchitectureCapability::LayerNormalization,
    });
}

fn mixer(mixer: &MixerSpec, capabilities: &mut BTreeSet<ArchitectureCapability>) {
    let output = match mixer {
        MixerSpec::SoftmaxAttention(attention) => {
            capabilities.insert(ArchitectureCapability::SoftmaxAttention);
            if attention.window.is_some() {
                capabilities.insert(ArchitectureCapability::SlidingWindowAttention);
            }
            if attention.sinks {
                capabilities.insert(ArchitectureCapability::AttentionSinks);
            }
            if attention.qk_normalization == QkNormalization::QueryKeyRms {
                capabilities.insert(ArchitectureCapability::QueryKeyRmsNormalization);
            }
            if attention.key_value_relation == crate::semantic::KeyValueRelation::KeyEqualsValue {
                capabilities.insert(ArchitectureCapability::SharedKeyValueProjection);
            }
            position(&attention.position, capabilities);
            attention.output
        },
        MixerSpec::LinearAttention(attention) => {
            capabilities.insert(ArchitectureCapability::LinearAttention);
            attention.output
        },
    };
    if output == AttentionOutputSpec::Gated {
        capabilities.insert(ArchitectureCapability::GatedAttentionOutput);
    }
}

fn position(position: &PositionEncodingSpec, capabilities: &mut BTreeSet<ArchitectureCapability>) {
    let PositionEncodingSpec::Rotary(rotary) = position else {
        return;
    };
    capabilities.insert(ArchitectureCapability::RotaryPositionEncoding);
    match rotary.layout {
        RotaryLayoutSpec::Standard => {},
        RotaryLayoutSpec::MultiSection(_) => {
            capabilities.insert(ArchitectureCapability::MultiSectionRotary);
        },
        RotaryLayoutSpec::InterleavedMultiSection(_) => {
            capabilities.insert(ArchitectureCapability::InterleavedMultiSectionRotary);
        },
    }
    match rotary.scaling {
        Some(RopeScalingSpec::PiecewiseFrequency { .. }) => {
            capabilities.insert(ArchitectureCapability::PiecewiseRopeScaling);
        },
        Some(RopeScalingSpec::Yarn { .. }) => {
            capabilities.insert(ArchitectureCapability::YarnRopeScaling);
        },
        None => {},
    }
}

fn feed_forward(
    feed_forward: &FeedForwardSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    match feed_forward {
        FeedForwardSpec::Dense { activation, .. } => {
            capabilities.insert(ArchitectureCapability::DenseFeedForward);
            activation_capability(activation, capabilities);
        },
        FeedForwardSpec::Routed { routed, shared } => {
            routed_capabilities(routed, capabilities);
            if let Some(shared) = shared {
                capabilities.insert(ArchitectureCapability::SharedExpert);
                activation_capability(&shared.activation, capabilities);
            }
        },
        FeedForwardSpec::DenseAndRouted { dense_activation, routed, .. } => {
            capabilities.insert(ArchitectureCapability::DenseFeedForward);
            capabilities.insert(ArchitectureCapability::DenseAndRouted);
            activation_capability(dense_activation, capabilities);
            routed_capabilities(routed, capabilities);
        },
    }
}

fn routed_capabilities(
    routed: &crate::semantic::RoutedExpertsSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    capabilities.insert(ArchitectureCapability::RoutedExperts);
    capabilities.insert(match routed.router_normalization {
        RouterNormalization::SoftmaxTopK => ArchitectureCapability::SoftmaxTopKRouter,
        RouterNormalization::UnitTopK => ArchitectureCapability::UnitTopKRouter,
    });
    activation_capability(&routed.activation, capabilities);
}

fn activation_capability(
    activation: &ActivationSpec,
    capabilities: &mut BTreeSet<ArchitectureCapability>,
) {
    capabilities.insert(match activation {
        ActivationSpec::SwiGlu { clamp: Some(_), .. } => ArchitectureCapability::ClampedSwiGlu,
        ActivationSpec::SwiGlu { clamp: None, .. } => ArchitectureCapability::SwiGlu,
        ActivationSpec::GeluTanh => ArchitectureCapability::GeluTanh,
        ActivationSpec::NamedGated { .. } => ArchitectureCapability::NamedGatedActivation,
    });
}
