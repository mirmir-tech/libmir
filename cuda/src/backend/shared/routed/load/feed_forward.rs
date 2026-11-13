use models::{
    layout::DecoderConfig,
    weights::{MixedFeedForwardBindings, TensorCatalog},
};

use crate::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, CudaAffineSharedExpertMoe,
    CudaBackend, CudaTensorSet, Error, GatedActivation, Result,
    backend::feed_forward::{DenseFeedForward, FeedForward},
};

pub(super) fn load(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    bindings: MixedFeedForwardBindings<'_>,
) -> Result<FeedForward> {
    match bindings {
        MixedFeedForwardBindings::Dense(bindings) => {
            DenseFeedForward::load(backend, decoder, tensors, bindings)
                .map(Box::new)
                .map(FeedForward::Dense)
        },
        MixedFeedForwardBindings::SharedRouted(bindings) => {
            let experts = decoder.num_experts.ok_or_else(|| {
                Error::UnsupportedDecoderLayer("missing parsed expert count".into())
            })?;
            let routed = decoder.moe_intermediate_size.ok_or_else(|| {
                Error::UnsupportedDecoderLayer("missing parsed routed expert width".into())
            })?;
            let weights = AffineSharedExpertMoeWeights::load_bindings(
                backend,
                tensors,
                catalog,
                bindings,
                experts,
                decoder.hidden_size,
                routed,
            )?;
            let config = moe_config(decoder, &weights)?;
            CudaAffineSharedExpertMoe::new(backend, config, weights)
                .map(Box::new)
                .map(FeedForward::SharedRouted)
        },
    }
}

fn moe_config(
    decoder: &DecoderConfig,
    weights: &AffineSharedExpertMoeWeights,
) -> Result<AffineSharedExpertMoeConfig> {
    let expert_count = decoder
        .num_experts
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing parsed expert count".into()))?;
    let top_k = decoder
        .top_k_experts
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing parsed expert top-k".into()))?;
    let routed = decoder.moe_intermediate_size.ok_or_else(|| {
        Error::UnsupportedDecoderLayer("missing parsed routed expert width".into())
    })?;
    let shared = decoder.shared_expert_intermediate_size.ok_or_else(|| {
        Error::UnsupportedDecoderLayer("missing parsed shared expert width".into())
    })?;
    let (group_size, expert_bits, router_bits) =
        weights.storage_format(expert_count, decoder.hidden_size, routed, shared)?;
    Ok(AffineSharedExpertMoeConfig {
        hidden_size: decoder.hidden_size,
        routed_intermediate_size: routed,
        shared_intermediate_size: shared,
        expert_count,
        top_k,
        group_size,
        expert_bits,
        router_bits,
        activation: GatedActivation::try_from(decoder)?,
    })
}
