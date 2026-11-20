//! `DeepSeek` feed-forwards: dense in the first layers, then routed experts
//! plus always-active shared experts, weighted as the semantic router says.

use models::{
    layout::DecoderConfig,
    semantic::{FeedForwardSpec, RouterNormalization},
    weights::{MixedFeedForwardBindings, TensorCatalog},
};

use crate::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, CudaAffineSharedExpertMoe,
    CudaBackend, CudaTensorSet, Error, GatedActivation, Result,
    backend::feed_forward::{DenseFeedForward, FeedForward},
    kernels::{RouterUnitScoring, RouterUnitWeighting},
};

pub(super) fn load(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    spec: &FeedForwardSpec,
    bindings: MixedFeedForwardBindings<'_>,
) -> Result<FeedForward> {
    match (bindings, spec) {
        (MixedFeedForwardBindings::Dense(bindings), FeedForwardSpec::Dense { .. }) => {
            DenseFeedForward::load(backend, decoder, tensors, bindings)
                .map(Box::new)
                .map(FeedForward::Dense)
        },
        (
            MixedFeedForwardBindings::SharedRouted(bindings),
            FeedForwardSpec::Routed { routed, shared: Some(shared) },
        ) => {
            let weights = AffineSharedExpertMoeWeights::load_bindings(
                backend,
                tensors,
                catalog,
                bindings,
                routed.expert_count,
                decoder.hidden_size,
                routed.intermediate_size,
            )?;
            let (group_size, expert_bits, router_bits) = weights.storage_format(
                routed.expert_count,
                decoder.hidden_size,
                routed.intermediate_size,
                shared.intermediate_size,
            )?;
            let config = AffineSharedExpertMoeConfig {
                hidden_size: decoder.hidden_size,
                routed_intermediate_size: routed.intermediate_size,
                shared_intermediate_size: shared.intermediate_size,
                expert_count: routed.expert_count,
                top_k: routed.top_k,
                group_size,
                expert_bits,
                router_bits,
                activation: GatedActivation::try_from(decoder)?,
                routing: scoring(routed.router_normalization, routed.output_scale)?,
            };
            CudaAffineSharedExpertMoe::new(backend, config, weights)
                .map(Box::new)
                .map(FeedForward::SharedRouted)
        },
        _ => Err(Error::UnsupportedDecoderLayer(
            "latent-attention feed-forward is neither dense nor routed with shared experts".into(),
        )),
    }
}

fn scoring(normalization: RouterNormalization, output_scale: f64) -> Result<RouterUnitScoring> {
    let weighting = match normalization {
        RouterNormalization::SoftmaxTopK => RouterUnitWeighting::Renormalized,
        RouterNormalization::SoftmaxProbabilityTopK => RouterUnitWeighting::Probability,
        RouterNormalization::SigmoidBiasedTopK => RouterUnitWeighting::SigmoidBiased,
        RouterNormalization::UnitTopK => {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA latent-attention models have no {normalization:?} router yet"
            )));
        },
    };
    #[allow(clippy::cast_possible_truncation)]
    let scale = output_scale as f32;
    Ok(RouterUnitScoring { weighting, scale })
}
