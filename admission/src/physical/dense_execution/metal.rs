//! Whether Metal runs a dense (BF16/F16/F32) checkpoint's generation: the
//! layers resolve in the binding plan of the runtime admission selects, the
//! plan that runtime's loader reads them by, with routed experts in a
//! layout that loader builds its banks from.

use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    semantic::{DecoderLayerSpec, NormPlacement},
    weights::{
        DenseLayerFeedForwardBindings, MixedFeedForwardBindings, RoutedExpertBindings,
        WeightBindingPlan,
    },
};

use crate::metal::{
    MetalArchitecture, MetalDecoderRuntime, RoutedExpertLoader, hybrid_experts_load,
};

pub(super) fn metal_generation_supported(
    task: &TaskExecutionPlan,
    contract: &DecoderExecutionContract,
) -> bool {
    let layers = &contract.semantic.decoder.layers;
    // Without routed experts dense storage loads as every runtime's plain
    // path; an architecture Metal does not run is the architecture check's
    // refusal, not this one's.
    if layers.iter().all(|layer| layer.feed_forward.routed().is_none()) {
        return true;
    }
    let Ok(MetalArchitecture::Generation(plan)) =
        crate::metal::admit(task, Some(&contract.semantic))
    else {
        return false;
    };
    let runtime = plan.runtime();
    layers.iter().all(|layer| layer_resolves(runtime, &contract.bindings, layer))
}

/// Whether `layer` resolves in `runtime`'s binding plan, as its loader in
/// `metal/src/native/model/load/backend.rs` reads it. A non-routed layer
/// (`Dense` or `None`) is every runtime's plain path; routed layers are read
/// by the runtime's own plan, their experts by the runtime's expert loader.
fn layer_resolves(
    runtime: MetalDecoderRuntime,
    bindings: &WeightBindingPlan,
    layer: &DecoderLayerSpec,
) -> bool {
    if layer.feed_forward.routed().is_none() {
        return true;
    }
    let index = layer.index;
    let experts = match runtime {
        // `HybridMoeModel`'s dense-plus-routed layers.
        MetalDecoderRuntime::DenseAndRouted | MetalDecoderRuntime::DenseGelu => {
            return bindings
                .hybrid_moe_layer(index)
                .is_ok_and(|bindings| hybrid_experts_load(&bindings.experts));
        },
        // Routed experts without sinks (Qwen3-MoE, Mixtral, Llama 4,
        // GLM-4-MoE): `DenseSwiGluModel`.
        MetalDecoderRuntime::Dense => dense_feed_forward(bindings, layer.norm_placement, index)
            .and_then(|feed_forward| match feed_forward {
                DenseLayerFeedForwardBindings::Gated(_) => None,
                DenseLayerFeedForwardBindings::Routed(routed) => Some(routed.experts),
                DenseLayerFeedForwardBindings::SharedRouted(shared) => Some(shared.routed),
            }),
        // `HybridLinearMoeModel`, by layer: linear or softmax attention with
        // shared experts, or latent attention.
        MetalDecoderRuntime::SharedRouted | MetalDecoderRuntime::DenseMixed => bindings
            .mixed_decoder_layer(index)
            .ok()
            .and_then(|layer| mixed(layer.feed_forward)),
        MetalDecoderRuntime::LatentAttention => bindings
            .latent_decoder_layer(index)
            .ok()
            .and_then(|layer| mixed(layer.feed_forward)),
        MetalDecoderRuntime::ClampedRouted => {
            bindings.routed_decoder_layer(index).ok().map(|layer| layer.experts)
        },
        MetalDecoderRuntime::StateSpaceHybrid => bindings
            .state_space_hybrid_layer(index)
            .ok()
            .and_then(|layer| layer.feed_forward?.routed())
            .map(|routed| routed.routed),
    };
    experts.is_some_and(|experts| loads(runtime, experts))
}

fn dense_feed_forward(
    bindings: &WeightBindingPlan,
    placement: NormPlacement,
    index: usize,
) -> Option<DenseLayerFeedForwardBindings<'_>> {
    match placement {
        NormPlacement::Pre => bindings.dense_decoder_layer(index).ok().map(|l| l.feed_forward),
        NormPlacement::Post => bindings.post_norm_decoder_layer(index).ok().map(|l| l.feed_forward),
        NormPlacement::Parallel => {
            bindings.parallel_decoder_layer(index).ok().map(|l| l.feed_forward)
        },
    }
}

const fn mixed(feed_forward: MixedFeedForwardBindings<'_>) -> Option<RoutedExpertBindings<'_>> {
    match feed_forward {
        MixedFeedForwardBindings::Dense(_) => None,
        MixedFeedForwardBindings::SharedRouted(shared) => Some(shared.routed),
    }
}

fn loads(runtime: MetalDecoderRuntime, experts: RoutedExpertBindings<'_>) -> bool {
    RoutedExpertLoader::of(runtime).is_some_and(|loader| loader.loads(experts))
}
