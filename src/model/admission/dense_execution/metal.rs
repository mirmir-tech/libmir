//! Whether Metal runs a dense (BF16/F16/F32) checkpoint's generation: the
//! layers resolve in the binding plan of the runtime admission selects, the
//! plan that runtime's loader reads them by.

use architecture::metal::{MetalArchitecture, MetalDecoderRuntime};
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    semantic::{DecoderLayerSpec, FeedForwardSpec},
    weights::{HybridMoeExpertBindings, TensorStorage, WeightBindingPlan},
};

pub(super) fn metal_generation_supported(
    task: &TaskExecutionPlan,
    contract: &DecoderExecutionContract,
) -> bool {
    let Ok(MetalArchitecture::Generation(plan)) =
        architecture::metal::admit(task, Some(&contract.semantic))
    else {
        return false;
    };
    let runtime = plan.runtime();
    contract
        .semantic
        .decoder
        .layers
        .iter()
        .all(|layer| layer_resolves(runtime, &contract.bindings, layer))
}

/// Whether `layer` resolves in `runtime`'s binding plan, as its loader in
/// `metal/src/native/model/load/backend.rs` reads it. A dense layer is
/// every runtime's plain path; routed layers are read by the runtime's own
/// plan.
fn layer_resolves(
    runtime: MetalDecoderRuntime,
    bindings: &WeightBindingPlan,
    layer: &DecoderLayerSpec,
) -> bool {
    if matches!(layer.feed_forward, FeedForwardSpec::Dense { .. }) {
        return true;
    }
    let index = layer.index;
    match runtime {
        // Routed experts without sinks (Qwen3-MoE, Mixtral, Llama 4,
        // GLM-4-MoE): `DenseSwiGluModel`.
        MetalDecoderRuntime::Dense => bindings.dense_decoder_layer(index).is_ok(),
        // `HybridMoeModel`'s dense-plus-routed layers.
        MetalDecoderRuntime::DenseAndRouted | MetalDecoderRuntime::DenseGelu => bindings
            .hybrid_moe_layer(index)
            .is_ok_and(|bindings| hybrid_experts_supported(&bindings.experts)),
        // `HybridLinearMoeModel`, by layer: linear or softmax attention with
        // shared experts, or latent attention.
        MetalDecoderRuntime::SharedRouted | MetalDecoderRuntime::DenseMixed => {
            bindings.mixed_decoder_layer(index).is_ok()
        },
        MetalDecoderRuntime::LatentAttention => bindings.latent_decoder_layer(index).is_ok(),
        MetalDecoderRuntime::ClampedRouted => bindings.routed_decoder_layer(index).is_ok(),
        MetalDecoderRuntime::StateSpaceHybrid => bindings.state_space_hybrid_layer(index).is_ok(),
    }
}

fn hybrid_experts_supported(experts: &HybridMoeExpertBindings<'_>) -> bool {
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
