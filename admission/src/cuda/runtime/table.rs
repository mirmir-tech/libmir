//! What each CUDA runtime executes, one exhaustive `match`.

use models::semantic::QkNormalization;

use super::CudaDecoderRuntime;
use crate::{
    capability::{
        ActivationKind::{ClampedSwiGlu, GeluTanh, ReluSquared, SwiGlu},
        ActivationSite::{Dense, Expert, Shared},
        FeedForwardShape as Block, MixerKind, Operations, RouterKind,
    },
    decoder::Feature,
    set::Flag,
};

/// No CUDA runtime executes Gemma 4's E-model layer sharing.
const UNSUPPORTED: [Feature; 2] = [Feature::PerLayerInput, Feature::SharedLayerKeyValues];
/// Only the graph decoder executes these, beside what none executes.
const GRAPH_ONLY: [Feature; 9] = [
    Feature::PerLayerInput,
    Feature::SharedLayerKeyValues,
    Feature::ResidualScaling,
    Feature::ScoreSoftcap,
    Feature::PostOrParallelNorm,
    Feature::LayerNorm,
    Feature::ChunkedAttention,
    Feature::QueryTemperature,
    Feature::AttentionFree,
];
/// Q/K norms outside the graph decoder.
const NORMALIZATIONS: [QkNormalization; 3] =
    [QkNormalization::None, QkNormalization::QueryKeyRms, QkNormalization::QueryKeyValueRms];

pub(super) fn capabilities(runtime: CudaDecoderRuntime) -> Operations {
    let none = Operations::default();
    let softmax = MixerKind::SOFTMAX;
    let mixed = std::iter::once(MixerKind::Linear).chain(softmax);
    match runtime {
        CudaDecoderRuntime::Dense => none
            .layers(softmax, &[Block::Dense])
            .layers(MixerKind::SOFTMAX_WITHOUT_SINKS, &[Block::Routed, Block::SharedRouted])
            .activations(Dense, &[SwiGlu, GeluTanh])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::Routed, Block::SharedRouted], <RouterKind as Flag>::ALL)
            .normalizations(<QkNormalization as Flag>::ALL)
            .features_except(&[&UNSUPPORTED[..], &[Feature::AttentionFree]].concat()),
        CudaDecoderRuntime::DenseAndRouted => none
            .layers(softmax, &[Block::DenseAndRouted])
            .activations(Dense, &[SwiGlu, GeluTanh])
            .activations(Expert, &[SwiGlu, GeluTanh])
            .routers(&[Block::DenseAndRouted], &RouterKind::TOP_K)
            .normalizations(&NORMALIZATIONS)
            .features_except(&GRAPH_ONLY),
        CudaDecoderRuntime::SharedRouted => none
            .layers(mixed, &[Block::SharedRouted])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .normalizations(&NORMALIZATIONS)
            .features_except(&GRAPH_ONLY),
        CudaDecoderRuntime::DenseMixed => none
            .layers(mixed, &[Block::Dense])
            .activations(Dense, &[SwiGlu, GeluTanh])
            .normalizations(&NORMALIZATIONS)
            .features_except(&GRAPH_ONLY),
        // The sink-attention runtime renormalizes a softmax over the selected
        // experts, without groups or an output factor, behind plain attention
        // with separate values.
        CudaDecoderRuntime::ClampedRouted => none
            .layers(MixerKind::SOFTMAX_WITH_SINKS, &[Block::Routed])
            .activations(Expert, &[ClampedSwiGlu])
            .routers(&[Block::Routed], &[RouterKind::SoftmaxTopK])
            .normalizations(&[QkNormalization::None])
            .features_except(
                &[
                    &GRAPH_ONLY[..],
                    &[Feature::ExpertGroups, Feature::ExpertOutputScale, Feature::SharedKeyValue],
                ]
                .concat(),
            ),
        // Plain attention beside the Mamba-2 mixers: no sinks, Q/K norm or
        // positions, separate values; biased sigmoid routing into relu² experts.
        CudaDecoderRuntime::StateSpaceHybrid => {
            let latent = [Block::LatentRouted, Block::SharedLatentRouted];
            none.layers(
                std::iter::once(MixerKind::StateSpace).chain(MixerKind::SOFTMAX_WITHOUT_SINKS),
                &[Block::LatentRouted, Block::SharedLatentRouted, Block::Dense, Block::None],
            )
            .activations(Dense, &[ReluSquared])
            .activations(Expert, &[ReluSquared])
            .activations(Shared, &[ReluSquared])
            .routers(&latent, &[RouterKind::SigmoidBiasedTopK])
            .normalizations(&[QkNormalization::None])
            .features_except(
                &[&GRAPH_ONLY[..], &[Feature::SharedKeyValue, Feature::RotaryAttention]].concat(),
            )
        },
        CudaDecoderRuntime::LatentAttention => none
            .layers([MixerKind::Latent], &[Block::Dense, Block::SharedRouted])
            .activations(Dense, &[SwiGlu])
            .activations(Expert, &[SwiGlu])
            .activations(Shared, &[SwiGlu])
            .routers(&[Block::SharedRouted], &RouterKind::TOP_K)
            .features_except(&GRAPH_ONLY),
    }
}
