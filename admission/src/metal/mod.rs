//! Metal architecture admission: lower, require, select.

mod experts;
mod lowering;
mod runtime;

pub use experts::{IndividualBank, RoutedExpertLoader, hybrid_experts_load};
pub use lowering::{LayerLowering, NormalizationLowering, lower_layer};
use models::{
    execution::{PoolingMode, TaskExecutionPlan},
    layout::{
        DecoderConfig, EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling,
        HiddenActivation, NormKind, VisionPipeline,
    },
    semantic::SemanticModelSpec,
};
pub use runtime::MetalDecoderRuntime;

pub use crate::lowering::{Activation, Clamp, Experts, FeedForwardLowering, MixerLowering};
use crate::{
    ArchitectureError, CacheLayout, Result, TaskRefusal, capability::Operations,
    decoder::DecoderFeatures,
};

/// A lowered decoder and the runtime admitted to execute it, which the
/// loaders build from.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalDecoderPlan {
    layers: Vec<LayerLowering>,
    final_norm: NormalizationLowering,
    features: DecoderFeatures,
    runtime: MetalDecoderRuntime,
    cache_layout: CacheLayout,
}

impl MetalDecoderPlan {
    #[must_use]
    pub fn layers(&self) -> &[LayerLowering] {
        &self.layers
    }

    #[must_use]
    pub const fn final_norm(&self) -> NormalizationLowering {
        self.final_norm
    }

    #[must_use]
    pub const fn features(&self) -> DecoderFeatures {
        self.features
    }

    #[must_use]
    pub const fn runtime(&self) -> MetalDecoderRuntime {
        self.runtime
    }

    #[must_use]
    pub const fn cache_layout(&self) -> &CacheLayout {
        &self.cache_layout
    }

    /// What the plan requires of a runtime.
    #[must_use]
    pub fn requirements(&self) -> Operations {
        Operations::required(
            self.layers.iter().map(|layer| (&layer.mixer, &layer.feed_forward)),
            self.features,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetalArchitecture {
    Generation(MetalDecoderPlan),
    Embedding,
    SequenceScoring,
}

impl MetalArchitecture {
    /// The admitted generation runtime.
    #[must_use]
    pub const fn runtime(&self) -> Option<MetalDecoderRuntime> {
        match self {
            Self::Generation(plan) => Some(plan.runtime),
            Self::Embedding | Self::SequenceScoring => None,
        }
    }
}

/// Whether Metal runs a vision tower for `pipeline`. A model with another
/// pipeline loads text-only, and its image requests fail.
#[must_use]
pub const fn executes_vision(pipeline: VisionPipeline) -> bool {
    match pipeline {
        VisionPipeline::PooledEncoder
        | VisionPipeline::SpatialMergeEncoder
        | VisionPipeline::PixelShuffleEncoder => true,
    }
}

pub fn admit(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<MetalArchitecture> {
    match task {
        TaskExecutionPlan::Generation { decoder }
        | TaskExecutionPlan::CausalScoring { decoder, .. } => {
            let semantic = semantic.ok_or(ArchitectureError::MissingSemanticContract)?;
            plan(decoder, semantic).map(MetalArchitecture::Generation)
        },
        TaskExecutionPlan::Embedding { task, .. }
            if task.pooling == PoolingMode::LastToken && task.normalize =>
        {
            Ok(MetalArchitecture::Embedding)
        },
        TaskExecutionPlan::Embedding { .. } => {
            Err(ArchitectureError::Task(TaskRefusal::EmbeddingPooling))
        },
        TaskExecutionPlan::SequenceScoring { encoder, .. } => sequence_scoring(encoder),
    }
}

/// Lowers `spec`, whose configuration is `decoder`, and selects its runtime.
pub fn plan(decoder: &DecoderConfig, spec: &SemanticModelSpec) -> Result<MetalDecoderPlan> {
    let lowered = lowering::Lowered::lower(spec, decoder.attention_scale)?;
    let runtime = MetalDecoderRuntime::select(&lowered.requirements())?;
    Ok(MetalDecoderPlan {
        layers: lowered.layers,
        final_norm: lowered.final_norm,
        features: lowered.features,
        runtime,
        cache_layout: CacheLayout::for_metal(&spec.decoder),
    })
}

fn sequence_scoring(config: &EncoderConfig) -> Result<MetalArchitecture> {
    let rope =
        matches!(config.rope_scaling, None | Some(EncoderRopeScaling::Ntk { mixed_b: None, .. }));
    if config.packed_qkv
        && config.norm == NormKind::LayerNorm
        && config.hidden_activation == HiddenActivation::Gelu
        && config.position_embedding == EncoderPositionEmbedding::Rope
        && config.num_labels == 1
        && rope
    {
        Ok(MetalArchitecture::SequenceScoring)
    } else {
        Err(ArchitectureError::Task(TaskRefusal::SequenceScoringEncoder))
    }
}
