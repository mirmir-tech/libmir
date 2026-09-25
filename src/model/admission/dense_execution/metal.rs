use models::{
    execution::DecoderExecutionContract,
    semantic::FeedForwardSpec,
    weights::{HybridMoeExpertBindings, TensorStorage},
};

pub(super) fn metal_generation_supported(contract: &DecoderExecutionContract) -> bool {
    contract.semantic.decoder.layers.iter().all(|layer| match &layer.feed_forward {
        FeedForwardSpec::Dense { .. } => true,
        FeedForwardSpec::DenseAndRouted { .. } => contract
            .bindings
            .hybrid_moe_layer(layer.index)
            .is_ok_and(|bindings| metal_hybrid_experts_supported(&bindings.experts)),
        FeedForwardSpec::Routed { shared: Some(_), .. } => {
            contract.bindings.hybrid_decoder_layer(layer.index).is_ok()
        },
        FeedForwardSpec::Routed { shared: None, .. } => {
            contract.bindings.routed_decoder_layer(layer.index).is_ok()
        },
    })
}

fn metal_hybrid_experts_supported(experts: &HybridMoeExpertBindings<'_>) -> bool {
    match experts {
        HybridMoeExpertBindings::Stacked(_) | HybridMoeExpertBindings::FusedStacked { .. } => true,
        HybridMoeExpertBindings::Individual { gate, up, down } => {
            gate.iter().chain(up).chain(down).all(|binding| {
                matches!(
                    binding.storage,
                    TensorStorage::BlockQuantized {
                        format: models::weights::BlockQuantization::NVFP4,
                        ..
                    }
                )
            })
        },
    }
}
