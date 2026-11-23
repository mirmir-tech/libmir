//! A mixed decoder layer's feed-forward from its semantic spec: a dense
//! gated MLP, or routed experts plus always-active shared experts, weighted
//! as the semantic router says.

use models::{
    semantic::FeedForwardSpec,
    weights::{MixedFeedForwardBindings, TensorCatalog},
};

use crate::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, CudaAffineSharedExpertMoe,
    CudaBackend, CudaTensorSet, Error, GatedActivation, Result,
    backend::feed_forward::{DenseFeedForward, FeedForward},
    kernels::RouterUnitScoring,
};

pub(in crate::backend) fn load(
    backend: &CudaBackend,
    hidden_size: usize,
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    spec: &FeedForwardSpec,
    bindings: MixedFeedForwardBindings<'_>,
) -> Result<FeedForward> {
    match (bindings, spec) {
        (
            MixedFeedForwardBindings::Dense(bindings),
            FeedForwardSpec::Dense { intermediate_size, activation },
        ) => DenseFeedForward::load(
            backend,
            (hidden_size, *intermediate_size),
            GatedActivation::try_from(activation)?,
            tensors,
            bindings,
        )
        .map(Box::new)
        .map(FeedForward::Dense),
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
                hidden_size,
                routed.intermediate_size,
            )?;
            let (group_size, expert_bits, router_bits) = weights.storage_format(
                routed.expert_count,
                hidden_size,
                routed.intermediate_size,
                shared.intermediate_size,
            )?;
            let config = AffineSharedExpertMoeConfig {
                hidden_size,
                routed_intermediate_size: routed.intermediate_size,
                shared_intermediate_size: shared.intermediate_size,
                expert_count: routed.expert_count,
                top_k: routed.top_k,
                group_size,
                expert_bits,
                router_bits,
                activation: GatedActivation::try_from(&routed.activation)?,
                routing: RouterUnitScoring::of(routed)?,
            };
            CudaAffineSharedExpertMoe::new(backend, config, weights)
                .map(Box::new)
                .map(FeedForward::SharedRouted)
        },
        _ => Err(Error::UnsupportedDecoderLayer(
            "mixed-layer feed-forward is neither dense nor routed with shared experts".into(),
        )),
    }
}
