//! What each Metal runtime executes, one exhaustive `match`. Only the dense
//! runtime reads every declared constant; the others implement theirs in
//! the kernels and refuse a model that declares another.

use models::semantic::QkNormalization;

use super::MetalDecoderRuntime;
use crate::{
    capability::{
        ActivationKind::{ClampedSwiGlu, GeluTanh, ReluSquared, SwiGlu},
        ActivationSite::{Dense, Expert, Shared},
        FeedForwardShape as Block, MixerKind, Operations, RouterKind,
    },
    decoder::Feature,
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
    // Unit embeddings and the default `1 / sqrt(head_dim)` scale; the linear
    // mixers run without any softmax layer beside them.
    let plain = refusing(&[Feature::EmbeddingNotUnit, Feature::AttentionScaleNotDefault]);
    let plain = plain.iter().copied().filter(|&feature| feature != Feature::AttentionFree);
    let plain = plain.collect::<Vec<_>>();
    // Unit embeddings and the configuration's scale.
    let declared = refusing(&[Feature::EmbeddingNotUnit, Feature::AttentionScaleUndeclared]);
    // Gemma 4: `sqrt(hidden)` embeddings and a unit scale.
    let gemma = refusing(&[Feature::EmbeddingNotSqrtHidden, Feature::AttentionScaleNotUnit]);
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
            .features_except(&[
                Feature::AttentionFree,
                Feature::AttentionScaleUndeclared,
                Feature::ExpertGroups,
                Feature::ExpertOutputScale,
                Feature::PerLayerInput,
                Feature::SharedLayerKeyValues,
            ]),
        MetalDecoderRuntime::DenseGelu => {
            let gemma = gemma.iter().filter(|feature| !SHARING.contains(feature)).copied();
            none.layers(softmax, &[Block::Dense])
                .activations(Dense, &[GeluTanh])
                .normalizations(&[QkNormalization::QueryKeyValueRms])
                .features_except(&gemma.collect::<Vec<_>>())
        },
        MetalDecoderRuntime::DenseAndRouted => none
            .layers(softmax, &[Block::DenseAndRouted])
            .activations(Dense, &[GeluTanh])
            .activations(Expert, &[GeluTanh])
            .routers(&[Block::DenseAndRouted], &RouterKind::TOP_K)
            .normalizations(&[QkNormalization::QueryKeyValueRms])
            .features_except(&gemma),
        MetalDecoderRuntime::SharedRouted => none
            .layers(mixed, &[Block::SharedRouted])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .normalizations(&[QkNormalization::QueryKeyRms])
            .features_except(&plain),
        MetalDecoderRuntime::DenseMixed => none
            .layers(mixed, &[Block::Dense])
            .activations(Dense, &[SwiGlu])
            .normalizations(&[QkNormalization::QueryKeyRms])
            .features_except(&plain),
        MetalDecoderRuntime::ClampedRouted => none
            .layers(MixerKind::SOFTMAX_WITH_SINKS, &[Block::Routed])
            .activations(Expert, &[ClampedSwiGlu])
            .routers(&[Block::Routed], &RouterKind::TOP_K)
            .normalizations(&[QkNormalization::None])
            .features_except(&declared),
        MetalDecoderRuntime::StateSpaceHybrid => {
            let latent = [Block::LatentRouted, Block::SharedLatentRouted];
            none.layers(
                [MixerKind::StateSpace, MixerKind::Softmax],
                &[Block::LatentRouted, Block::SharedLatentRouted, Block::Dense, Block::None],
            )
            .activations(Dense, &[ReluSquared])
            .activations(Expert, &[ReluSquared])
            .activations(Shared, &[ReluSquared])
            .routers(&latent, &[RouterKind::SigmoidBiasedTopK])
            .normalizations(&[QkNormalization::None])
            .features_except(&declared)
        },
        MetalDecoderRuntime::LatentAttention => none
            .layers([MixerKind::Latent], &[Block::Dense, Block::SharedRouted])
            .activations(Dense, &[SwiGlu, GeluTanh])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .features_except(&refusing(&[Feature::EmbeddingNotUnit])),
    }
}
