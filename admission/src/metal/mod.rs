use models::{
    execution::{PoolingMode, TaskExecutionPlan},
    layout::{EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind},
    semantic::SemanticModelSpec,
};

use crate::{ArchitectureError, Result};

mod operations;

pub use operations::{
    FeedForwardLowering, LayerLowering, MixerLowering, NormalizationLowering, lower_layer,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalDecoderRuntime {
    Dense,
    DenseAndRouted,
    SharedRouted,
    ClampedRouted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDecoderPlan {
    layers: Vec<LayerLowering>,
    runtime: MetalDecoderRuntime,
}

impl MetalDecoderPlan {
    #[must_use]
    pub fn layers(&self) -> &[LayerLowering] {
        &self.layers
    }

    #[must_use]
    pub const fn runtime(&self) -> MetalDecoderRuntime {
        self.runtime
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalArchitecture {
    Generation(MetalDecoderRuntime),
    Embedding,
    SequenceScoring,
}

pub fn admit(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<MetalArchitecture> {
    match task {
        TaskExecutionPlan::Generation { .. } => {
            let semantic = semantic.ok_or_else(|| {
                ArchitectureError::invalid("Metal generation semantic contract is missing")
            })?;
            Ok(MetalArchitecture::Generation(plan(semantic)?.runtime()))
        },
        TaskExecutionPlan::Embedding { task, .. }
            if task.pooling == PoolingMode::LastToken && task.normalize =>
        {
            Ok(MetalArchitecture::Embedding)
        },
        TaskExecutionPlan::Embedding { .. } => Err(ArchitectureError::invalid(
            "Metal embedding requires normalized last-token pooling",
        )),
        TaskExecutionPlan::SequenceScoring { encoder, .. } => sequence_scoring(encoder),
    }
}

fn sequence_scoring(config: &EncoderConfig) -> Result<MetalArchitecture> {
    let rope =
        matches!(config.rope_scaling, None | Some(EncoderRopeScaling::Ntk { mixed_b: None, .. }));
    if config.packed_qkv
        && config.norm == NormKind::LayerNorm
        && config.hidden_activation == "gelu"
        && config.position_embedding == EncoderPositionEmbedding::Rope
        && config.num_labels == 1
        && rope
    {
        Ok(MetalArchitecture::SequenceScoring)
    } else {
        Err(ArchitectureError::invalid(
            "Metal sequence scoring requires packed QKV, LayerNorm, GELU, one label, and supported RoPE",
        ))
    }
}

pub fn plan(spec: &SemanticModelSpec) -> Result<MetalDecoderPlan> {
    let layers = spec.decoder.layers.iter().map(lower_layer).collect::<Result<Vec<_>>>()?;
    let runtime = select_runtime(&layers)?;
    Ok(MetalDecoderPlan { layers, runtime })
}

fn select_runtime(layers: &[LayerLowering]) -> Result<MetalDecoderRuntime> {
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::Dense
            && matches!(layer.mixer, MixerLowering::Softmax { window: None, .. })
    }) {
        return Ok(MetalDecoderRuntime::Dense);
    }
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::DenseAndRouted
            && matches!(layer.mixer, MixerLowering::Softmax { .. })
    }) {
        return Ok(MetalDecoderRuntime::DenseAndRouted);
    }
    if layers
        .iter()
        .all(|layer| layer.feed_forward == FeedForwardLowering::SharedRouted)
        && layers.iter().any(|layer| layer.mixer == MixerLowering::Linear)
    {
        return Ok(MetalDecoderRuntime::SharedRouted);
    }
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::ClampedRouted
            && matches!(layer.mixer, MixerLowering::Softmax { sinks: true, .. })
    }) {
        return Ok(MetalDecoderRuntime::ClampedRouted);
    }
    Err(ArchitectureError::invalid(
        "Metal has no runtime for the lowered decoder operation composition",
    ))
}
