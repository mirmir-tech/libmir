//! CUDA's dense decoder adds projection biases (Qwen2, GLM-4 attention) only
//! to GGML layers; its other projection formats have no bias epilogue, so
//! its loader refuses a biased layer (`checkpoint/dense/load/bias.rs`) and
//! admission refuses it first.

use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    weights::{GgmlCompanion, LayerTensorRole, LogicalTensorRole, TensorBinding, TensorStorage},
};

use crate::cuda::{CudaArchitecture, CudaDecoderRuntime};

/// The first biased projection CUDA's dense decoder would refuse to load.
pub(super) fn unapplied<'a>(
    execution: Option<&'a DecoderExecutionContract>,
    task: &TaskExecutionPlan,
) -> Option<&'a TensorBinding> {
    let contract = execution?;
    let Ok(CudaArchitecture::Generation(admitted)) =
        crate::cuda::admit(task, Some(&contract.semantic))
    else {
        return None;
    };
    // CUDA's projection format is GGML when the plan has GGML blocks and no
    // grouped affine storage, which it selects first.
    let ggml = contract.bindings.affine_group_size().is_none() && contract.bindings.uses_ggml();
    if admitted.runtime != CudaDecoderRuntime::Dense || ggml {
        return None;
    }
    contract
        .bindings
        .tensors
        .iter()
        .find(|binding| projection(binding) && biased(binding))
}

/// A decoder layer's projection, which the loader adds a bias to; norms pack
/// their biases with their weights instead.
const fn projection(binding: &TensorBinding) -> bool {
    matches!(
        binding.role,
        LogicalTensorRole::Layer {
            tensor: LayerTensorRole::AttentionProjection { .. }
                | LayerTensorRole::FeedForwardProjection { .. }
                | LayerTensorRole::ExpertProjection { .. }
                | LayerTensorRole::SharedExpertProjection { .. }
                | LayerTensorRole::SharedExpertOutputGate
                | LayerTensorRole::Router
                | LayerTensorRole::PerLayerInput {
                    tensor: models::weights::PerLayerInputRole::Gate
                        | models::weights::PerLayerInputRole::Projection,
                },
            ..
        }
    )
}

const fn biased(binding: &TensorBinding) -> bool {
    matches!(
        &binding.storage,
        TensorStorage::Dense { bias: Some(_), .. }
            | TensorStorage::Ggml { companion: GgmlCompanion::Bias(_), .. }
    )
}
