use models::{
    execution::TaskExecutionPlan,
    weights::{BlockFormat, ExpertProjectionLayout},
};
use runtime::{kv::CacheConfig, trace::TraceAction};

use super::LoadedModel;
use crate::engine::lowering::CudaDecoderPlan;

pub(super) fn acceleration(model: &LoadedModel) -> Vec<String> {
    if !matches!(&model.task_plan, TaskExecutionPlan::Generation { .. }) {
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
    let Some(plan) = decoder_plan(model) else {
        return "packed gated exact-GELU encoder feed-forward";
    };
    if plan.all_dense_and_routed() {
        "dense gated MLP plus routed NVFP4 experts"
    } else if plan.all_dense() && dense_nvfp4(model) {
        "native NVFP4 gate/up and down projections"
    } else if plan.all_dense() {
        "paired BF16 gate/up plus BF16 down projection"
    } else if plan.all_shared_routed() {
        "shared expert plus routed SwiGLU experts"
    } else if plan.all_unshared_clamped_routed() && clamped_routed_mlx(model) {
        "clamped routed SwiGLU over split MLX MXFP4 experts"
    } else if plan.all_unshared_clamped_routed() {
        "clamped routed SwiGLU over native MXFP4 blocks"
    } else {
        "composed semantic feed-forward operations"
    }
}

pub(super) fn warnings(model: &LoadedModel) -> Vec<String> {
    let Some(plan) = decoder_plan(model) else {
        return Vec::new();
    };
    if plan.all_dense_and_routed() {
        Vec::new()
    } else {
        vec![
            "CUDA continuous decode batching is unavailable for this lowered operation plan".into(),
        ]
    }
}

pub(super) fn actions(model: &LoadedModel, cache: CacheConfig) -> Vec<TraceAction> {
    if !matches!(&model.task_plan, TaskExecutionPlan::Generation { .. }) {
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
    let Some(plan) = decoder_plan(model) else {
        return "CUTLASS F16 encoder projections and native bidirectional attention";
    };
    if plan.all_dense_and_routed() {
        "CUTLASS NVFP4 routed-MoE kernels"
    } else if plan.all_dense() && dense_nvfp4(model) {
        "CUTLASS native NVFP4 dense SwiGLU kernels"
    } else if plan.all_dense() {
        "CUTLASS BF16 dense SwiGLU kernels"
    } else if plan.all_shared_routed() {
        "CUDA mixed softmax/linear attention with shared routed experts"
    } else if plan.all_unshared_clamped_routed() {
        "CUDA MXFP4, YaRN, sink-attention, and clamped SwiGLU kernels"
    } else {
        "CUDA composed semantic operators"
    }
}

fn decode_acceleration(model: &LoadedModel) -> &'static str {
    if decoder_plan(model)
        .is_some_and(|plan| plan.all_unshared_clamped_routed() || plan.all_shared_routed())
    {
        "device-resident per-layer operation execution"
    } else {
        "full-model CUDA Graph decode replay"
    }
}

fn decode_action(model: &LoadedModel) -> &'static str {
    if decoder_plan(model)
        .is_some_and(|plan| plan.all_unshared_clamped_routed() || plan.all_shared_routed())
    {
        "device-resident decode with token feedback"
    } else {
        "captured model replay with device token feedback"
    }
}

fn dense_nvfp4(model: &LoadedModel) -> bool {
    decoder_plan(model).is_some_and(|plan| plan.all_dense())
        && model
            .contract
            .as_ref()
            .is_some_and(|contract| contract.bindings.uses_block_format(BlockFormat::NvFp4))
}

fn decoder_plan(model: &LoadedModel) -> Option<CudaDecoderPlan> {
    model.semantic().map(CudaDecoderPlan::lower)
}

fn clamped_routed_mlx(model: &LoadedModel) -> bool {
    model.contract.as_ref().is_some_and(|contract| {
        contract.bindings.expert_projection_layout() == Some(ExpertProjectionLayout::SeparateGateUp)
    })
}
