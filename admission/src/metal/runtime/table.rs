//! What each Metal runtime executes, one exhaustive `match`. Only the dense
//! runtime reads every declared constant; the others implement theirs in
//! the kernels and refuse a model that declares another.

use models::semantic::QkNormalization;

use super::MetalDecoderRuntime;
use crate::{
    capability::{
        ActivationKind::{ClampedSwiGlu, GeluTanh, ReluSquared, SwiGlu},
        ActivationSite::{Dense, Expert, Shared},
        AttentionScaleValue, EmbeddingScale,
        EmbeddingScale::{SqrtHidden, Unit},
        FeedForwardShape as Block, MixerKind, Operations, RouterKind, ScaleSource,
    },
    decoder::Feature,
    set::Flag,
};

/// Gemma 4's E-model layer sharing, which only the gated-GELU runtime runs.
const SHARING: [Feature; 2] = [Feature::PerLayerInput, Feature::SharedLayerKeyValues];
/// Only the dense runtime executes these.
const DENSE_ONLY: [Feature; 9] = [
    Feature::ResidualScaling,
    Feature::LayerNorm,
    Feature::PostOrParallelNorm,
    Feature::ScoreSoftcap,
    Feature::LongRope,
    Feature::ChunkedAttention,
    Feature::QueryTemperature,
    Feature::NormWeightOffset,
    Feature::AttentionFree,
];

/// The dense-only features plus `extra`.
fn refusing(extra: &[Feature]) -> Vec<Feature> {
    [&DENSE_ONLY[..], &SHARING[..], extra].concat()
}

pub(super) fn capabilities(runtime: MetalDecoderRuntime) -> Operations {
    let none = Operations::default();
    let softmax = MixerKind::SOFTMAX;
    let mixed = std::iter::once(MixerKind::Linear).chain(softmax);
    // The linear mixers run without any softmax layer beside them.
    let plain = refusing(&[]);
    let plain = plain.iter().copied().filter(|&feature| feature != Feature::AttentionFree);
    let plain = plain.collect::<Vec<_>>();
    let gemma = refusing(&[]);
    let sources = <ScaleSource as Flag>::ALL;
    let values = <AttentionScaleValue as Flag>::ALL;
    // Unit embeddings and the default `1 / sqrt(head_dim)` scale.
    let plain_scales =
        |ops: Operations| ops.scales(&[Unit], &[AttentionScaleValue::InverseSqrtHeadDim], sources);
    // Unit embeddings and the configuration's scale.
    let declared_scales = |ops: Operations| ops.scales(&[Unit], values, &[ScaleSource::Configured]);
    // Gemma 4: `sqrt(hidden)` embeddings and a unit scale.
    let gemma_scales =
        |ops: Operations| ops.scales(&[SqrtHidden], &[AttentionScaleValue::Unit], sources);
    match runtime {
        MetalDecoderRuntime::Dense => none
            .layers(softmax, &[Block::Dense])
            .layers([MixerKind::Softmax], &[Block::Routed, Block::SharedRouted])
            .activations(Dense, &[SwiGlu, GeluTanh])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::Routed], &[RouterKind::SoftmaxTopK, RouterKind::SparseMixer])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .routers(&[Block::SharedRouted], &[RouterKind::SigmoidInputScaledTopK])
            .normalizations(&[
                QkNormalization::None,
                QkNormalization::QueryKeyRms,
                QkNormalization::QueryKeyRmsAcrossHeads,
                QkNormalization::QueryKeyUnitRms,
            ])
            .scales(<EmbeddingScale as Flag>::ALL, values, &[ScaleSource::Configured])
            .features_except(&[
                Feature::AttentionFree,
                Feature::ExpertGroups,
                Feature::ExpertOutputScale,
                Feature::PerLayerInput,
                Feature::SharedLayerKeyValues,
            ]),
        MetalDecoderRuntime::DenseGelu => {
            let gemma = gemma.iter().filter(|feature| !SHARING.contains(feature)).copied();
            gemma_scales(none)
                .layers(softmax, &[Block::Dense])
                .activations(Dense, &[GeluTanh])
                .normalizations(&[QkNormalization::QueryKeyValueRms])
                .features_except(&gemma.collect::<Vec<_>>())
        },
        MetalDecoderRuntime::DenseAndRouted => gemma_scales(none)
            .layers(softmax, &[Block::DenseAndRouted])
            .activations(Dense, &[GeluTanh])
            .activations(Expert, &[GeluTanh])
            .routers(&[Block::DenseAndRouted], &RouterKind::TOP_K)
            .normalizations(&[QkNormalization::QueryKeyValueRms])
            .features_except(&gemma),
        MetalDecoderRuntime::SharedRouted => plain_scales(none)
            .layers(mixed, &[Block::SharedRouted])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .normalizations(&[QkNormalization::QueryKeyRms])
            .features_except(&plain),
        MetalDecoderRuntime::DenseMixed => plain_scales(none)
            .layers(mixed, &[Block::Dense])
            .activations(Dense, &[SwiGlu])
            .normalizations(&[QkNormalization::QueryKeyRms])
            .features_except(&plain),
        MetalDecoderRuntime::ClampedRouted => declared_scales(none)
            .layers(MixerKind::SOFTMAX_WITH_SINKS, &[Block::Routed])
            .activations(Expert, &[ClampedSwiGlu])
            .routers(&[Block::Routed], &RouterKind::TOP_K)
            .normalizations(&[QkNormalization::None])
            .features_except(&refusing(&[])),
        MetalDecoderRuntime::StateSpaceHybrid => {
            let latent = [Block::LatentRouted, Block::SharedLatentRouted];
            declared_scales(none)
                .layers(
                    [MixerKind::StateSpace, MixerKind::Softmax],
                    &[Block::LatentRouted, Block::SharedLatentRouted, Block::Dense, Block::None],
                )
                .activations(Dense, &[ReluSquared])
                .activations(Expert, &[ReluSquared])
                .activations(Shared, &[ReluSquared])
                .routers(&latent, &[RouterKind::SigmoidBiasedTopK])
                .normalizations(&[QkNormalization::None])
                .features_except(&refusing(&[Feature::StateSpaceFree]))
        },
        MetalDecoderRuntime::LatentAttention => none
            .scales(&[Unit], values, sources)
            .layers([MixerKind::Latent], &[Block::Dense, Block::SharedRouted])
            .activations(Dense, &[SwiGlu, GeluTanh])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .features_except(&refusing(&[])),
    }
}
