use architecture::capability::FeedForwardShape;
use models::{
    execution::TaskExecutionPlan,
    weights::{
        BlockFormat, ExpertProjectionLayout, LayerTensorRole, LogicalTensorRole, TensorStorage,
    },
};
use runtime::{kv::CacheConfig, trace::TraceAction};

use super::LoadedModel;
use crate::{CudaAdmitted, CudaDecoderRuntime};

pub(super) fn acceleration(model: &LoadedModel) -> Vec<String> {
    if !matches!(
        &model.task_plan,
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
    ) {
        return vec![
            "safe Rust mircuda execution gateway".into(),
            "one explicit CUDA stream".into(),
            model_kernels(model).into(),
            "device-resident task output".into(),
        ];
    }
    let mut features = vec![
        "safe Rust mircuda execution gateway".into(),
        "one explicit CUDA stream".into(),
        model_kernels(model).into(),
        "device-resident paged KV cache".into(),
    ];
    features.push(decode_acceleration(model).into());
    features
}

pub(super) fn mlp_layout(model: &LoadedModel) -> &'static str {
    let Some(admitted) = &model.admitted else {
        return "packed gated exact-GELU encoder feed-forward";
    };
    let dense = every(admitted, FeedForwardShape::Dense);
    let shared = every(admitted, FeedForwardShape::SharedRouted);
    match admitted.runtime {
        CudaDecoderRuntime::DenseAndRouted if uses_nvfp4(model) => {
            "dense gated MLP plus routed NVFP4 experts"
        },
        CudaDecoderRuntime::DenseAndRouted => "dense gated MLP plus routed BF16 experts",
        CudaDecoderRuntime::Dense if dense && uses_nvfp4(model) => {
            "native NVFP4 gate/up and down projections"
        },
        CudaDecoderRuntime::Dense if dense => "paired BF16 gate/up plus BF16 down projection",
        CudaDecoderRuntime::Dense => "routed SwiGLU experts without a shared expert",
        CudaDecoderRuntime::ClampedRouted if routed_experts_dense(model) => {
            "clamped routed SwiGLU over BF16 experts"
        },
        CudaDecoderRuntime::ClampedRouted if clamped_routed_mlx(model) => {
            "clamped routed SwiGLU over split MLX MXFP4 experts"
        },
        CudaDecoderRuntime::ClampedRouted => "clamped routed SwiGLU over native MXFP4 blocks",
        _ if shared && routed_experts_dense(model) => {
            "BF16 shared expert plus routed BF16 SwiGLU experts"
        },
        _ if shared => "shared expert plus routed SwiGLU experts",
        _ => "composed semantic feed-forward operations",
    }
}

pub(super) fn warnings(model: &LoadedModel) -> Vec<String> {
    let Some(admitted) = &model.admitted else {
        return Vec::new();
    };
    if admitted.runtime == CudaDecoderRuntime::DenseAndRouted {
        Vec::new()
    } else {
        vec![
            "CUDA continuous decode batching is unavailable for this lowered operation plan".into(),
        ]
    }
}

pub(super) fn actions(model: &LoadedModel, cache: CacheConfig) -> Vec<TraceAction> {
    if !matches!(
        &model.task_plan,
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
    ) {
        return vec![
            TraceAction::new("inspect", "compiled task semantics from config and tensor schema"),
            TraceAction::new(
                "load",
                format!("uploaded {} tensors directly from safetensors", model.catalog.len()),
            ),
            TraceAction::new("execute", model_kernels(model)),
            TraceAction::new("output", "copied only the final task result from CUDA"),
        ];
    }
    vec![
        TraceAction::new(
            "inspect",
            format!(
                "compiled {} semantic decoder layers from config and tensor schema",
                model.semantic().map_or(0, |semantic| semantic.decoder.layers.len())
            ),
        ),
        TraceAction::new(
            "load",
            format!("uploaded {} tensors directly from safetensors", model.catalog.len()),
        ),
        TraceAction::new("prefill", "device-resident chunked causal model execution"),
        TraceAction::new("decode", decode_action(model)),
        TraceAction::new(
            "kv_cache",
            format!(
                "{} physical blocks of {} tokens using {}",
                cache.block_count, cache.block_size, cache.dtype
            ),
        ),
        TraceAction::new("sampling", "greedy and bounded top-k/top-p execute on CUDA"),
    ]
}

fn model_kernels(model: &LoadedModel) -> &'static str {
    if matches!(&model.task_plan, TaskExecutionPlan::Embedding { .. }) {
        return "CUTLASS BF16 projections with causal attention and device L2 normalization";
    }
    let Some(admitted) = &model.admitted else {
        return "CUTLASS F16 encoder projections and native bidirectional attention";
    };
    let dense = every(admitted, FeedForwardShape::Dense);
    let shared = every(admitted, FeedForwardShape::SharedRouted);
    match admitted.runtime {
        CudaDecoderRuntime::DenseAndRouted if uses_nvfp4(model) => {
            "CUTLASS NVFP4 routed-MoE kernels"
        },
        CudaDecoderRuntime::DenseAndRouted => "CUDA BF16 dense and selected-expert kernels",
        CudaDecoderRuntime::Dense if dense && uses_nvfp4(model) => {
            "CUTLASS native NVFP4 dense SwiGLU kernels"
        },
        CudaDecoderRuntime::Dense if dense => "CUTLASS BF16 dense SwiGLU kernels",
        CudaDecoderRuntime::Dense => "CUDA softmax attention with routed selected-expert kernels",
        CudaDecoderRuntime::ClampedRouted if routed_experts_dense(model) => {
            "CUDA BF16, YaRN, sink-attention, and clamped SwiGLU kernels"
        },
        CudaDecoderRuntime::ClampedRouted => {
            "CUDA MXFP4, YaRN, sink-attention, and clamped SwiGLU kernels"
        },
        _ if shared && routed_experts_dense(model) => {
            "CUDA mixed attention with BF16 shared and routed experts"
        },
        _ if shared => "CUDA mixed softmax/linear attention with shared routed experts",
        _ => "CUDA composed semantic operators",
    }
}

fn decode_acceleration(model: &LoadedModel) -> &'static str {
    if per_layer(model) {
        "device-resident per-layer operation execution"
    } else {
        "full-model CUDA Graph decode replay"
    }
}

fn decode_action(model: &LoadedModel) -> &'static str {
    if per_layer(model) {
        "device-resident decode with token feedback"
    } else {
        "captured model replay with device token feedback"
    }
}

fn uses_nvfp4(model: &LoadedModel) -> bool {
    model
        .contract
        .as_ref()
        .is_some_and(|contract| contract.bindings.uses_block_format(BlockFormat::NvFp4))
}

fn routed_experts_dense(model: &LoadedModel) -> bool {
    model.contract.as_ref().is_some_and(|contract| {
        contract.bindings.tensors.iter().any(|binding| {
            matches!(
                binding.role,
                LogicalTensorRole::Layer {
                    tensor: LayerTensorRole::ExpertProjection { .. },
                    ..
                }
            ) && matches!(binding.storage, TensorStorage::Dense { .. })
        })
    })
}

/// Whether every layer's feed-forward has `shape`.
fn every(admitted: &CudaAdmitted, shape: FeedForwardShape) -> bool {
    admitted.plan.layers().iter().all(|layer| layer.feed_forward.shape() == shape)
}

/// Decode runs layer by layer rather than as one captured graph.
fn per_layer(model: &LoadedModel) -> bool {
    model.admitted.as_ref().is_some_and(|admitted| {
        admitted.runtime == CudaDecoderRuntime::ClampedRouted
            || every(admitted, FeedForwardShape::SharedRouted)
    })
}

fn clamped_routed_mlx(model: &LoadedModel) -> bool {
    model.contract.as_ref().is_some_and(|contract| {
        contract.bindings.expert_projection_layout() == Some(ExpertProjectionLayout::SeparateGateUp)
    })
}
