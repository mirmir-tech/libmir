use models::{
    execution::{PoolingMode, TaskExecutionPlan},
    layout::{
        EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind, VisionPipeline,
    },
    semantic::SemanticModelSpec,
};

use crate::{ArchitectureError, Result};

mod operations;
mod semantics;

pub use operations::{
    FeedForwardLowering, LayerLowering, MixerLowering, NormalizationLowering, lower_layer,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalDecoderRuntime {
    Dense,
    DenseGelu,
    DenseAndRouted,
    SharedRouted,
    /// Dense `SwiGLU` layers mixing linear and softmax attention.
    DenseMixed,
    ClampedRouted,
    /// Mamba-2 and attention mixers with sigmoid-routed experts (Nemotron-H).
    StateSpaceHybrid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDecoderPlan {
    layers: Vec<LayerLowering>,
    final_norm: NormalizationLowering,
    runtime: MetalDecoderRuntime,
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
            let semantic = semantic.ok_or_else(|| {
                ArchitectureError::invalid("Metal generation semantic contract is missing")
            })?;
            let runtime = plan(semantic)?.runtime();
            semantics::validate(decoder, semantic, runtime)?;
            Ok(MetalArchitecture::Generation(runtime))
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
    let final_norm = operations::lower_normalization(
        spec.decoder.layers.len(),
        "final",
        spec.decoder.final_norm,
    )?;
    let runtime = select_runtime(&layers)?;
    Ok(MetalDecoderPlan { layers, final_norm, runtime })
}

fn select_runtime(layers: &[LayerLowering]) -> Result<MetalDecoderRuntime> {
    if layers.iter().any(|layer| layer.mixer == MixerLowering::Latent) {
        return Err(ArchitectureError::invalid(
            "Metal has no runtime for multi-head latent attention yet",
        ));
    }
    if layers.iter().all(|layer| match (layer.feed_forward, layer.mixer) {
        (FeedForwardLowering::Dense, MixerLowering::Softmax { .. }) => true,
        (FeedForwardLowering::Routed, MixerLowering::Softmax { window: None, sinks, .. }) => !sinks,
        _ => false,
    }) {
        return Ok(MetalDecoderRuntime::Dense);
    }
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::DenseGelu
            && matches!(layer.mixer, MixerLowering::Softmax { .. })
    }) {
        return Ok(MetalDecoderRuntime::DenseGelu);
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
    if layers.iter().all(|layer| layer.feed_forward == FeedForwardLowering::Dense)
        && layers.iter().any(|layer| layer.mixer == MixerLowering::Linear)
    {
        return Ok(MetalDecoderRuntime::DenseMixed);
    }
    if layers.iter().all(|layer| {
        layer.feed_forward == FeedForwardLowering::ClampedRouted
            && matches!(layer.mixer, MixerLowering::Softmax { sinks: true, .. })
    }) {
        return Ok(MetalDecoderRuntime::ClampedRouted);
    }
    if layers.iter().any(|layer| layer.mixer == MixerLowering::StateSpace)
        && layers.iter().any(|layer| matches!(layer.mixer, MixerLowering::Softmax { .. }))
        && layers.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                FeedForwardLowering::LatentRouted | FeedForwardLowering::None
            ) && matches!(
                layer.mixer,
                MixerLowering::StateSpace
                    | MixerLowering::Softmax { sinks: false, window: None, .. }
            )
        })
    {
        return Ok(MetalDecoderRuntime::StateSpaceHybrid);
    }
    Err(ArchitectureError::invalid(
        "Metal has no runtime for the lowered decoder operation composition",
    ))
}
