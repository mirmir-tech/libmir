use foundation::model::Quantization;
use models::execution::{DecoderArchetype, TaskExecutionPlan};
use runtime::{kv::CacheConfig, trace::TraceAction};

use super::LoadedModel;

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
    features.push("full-model CUDA Graph decode replay".into());
    features
}

pub(super) fn mlp_layout(model: &LoadedModel) -> &'static str {
    let Some(plan) = model.plan.as_ref() else {
        return "packed gated exact-GELU encoder feed-forward";
    };
    match (plan.decoder, dense_nvfp4(model)) {
        (DecoderArchetype::HybridMoe, _) => "dense gated MLP plus routed NVFP4 experts",
        (DecoderArchetype::DenseSwiGlu, true) => "native NVFP4 gate/up and down projections",
        (DecoderArchetype::DenseSwiGlu, false) => "paired BF16 gate/up plus BF16 down projection",
        (DecoderArchetype::HybridLinearMoe, _) => "shared expert plus routed SwiGLU experts",
    }
}

pub(super) fn warnings(model: &LoadedModel) -> Vec<String> {
    let Some(plan) = model.plan.as_ref() else {
        return Vec::new();
    };
    match plan.decoder {
        DecoderArchetype::DenseSwiGlu => {
            vec!["CUDA dense continuous decode batching is not yet enabled".into()]
        },
        DecoderArchetype::HybridLinearMoe => {
            vec!["CUDA gated-delta execution is not yet implemented".into()]
        },
        DecoderArchetype::HybridMoe => Vec::new(),
    }
}

pub(super) fn actions(model: &LoadedModel, cache: CacheConfig) -> Vec<TraceAction> {
    if !matches!(&model.task_plan, TaskExecutionPlan::Generation { .. }) {
        return vec![
            TraceAction::new(
                "inspect",
                format!("recognized {:?} from config and tensor schema", model.metadata.family),
            ),
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
            format!("recognized {:?} from config and tensor schema", model.metadata.family),
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
    let Some(plan) = model.plan.as_ref() else {
        return "CUTLASS F16 encoder projections and native bidirectional attention";
    };
    match (plan.decoder, dense_nvfp4(model)) {
        (DecoderArchetype::HybridMoe, _) => "CUTLASS NVFP4 routed-MoE kernels",
        (DecoderArchetype::DenseSwiGlu, true) => "CUTLASS native NVFP4 dense SwiGLU kernels",
        (DecoderArchetype::DenseSwiGlu, false) => "CUTLASS BF16 dense SwiGLU kernels",
        (DecoderArchetype::HybridLinearMoe, _) => "CUDA hybrid linear-MoE kernels",
    }
}

const fn decode_action(_model: &LoadedModel) -> &'static str {
    "captured model replay with device token feedback"
}

fn dense_nvfp4(model: &LoadedModel) -> bool {
    model
        .plan
        .as_ref()
        .is_some_and(|plan| plan.decoder == DecoderArchetype::DenseSwiGlu)
        && matches!(&model.manifest.quantization, Quantization::NvFp4)
}
