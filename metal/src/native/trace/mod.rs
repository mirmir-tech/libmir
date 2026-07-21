mod description;
mod kv;
mod shape;

use description::{tensors as tensor_trace, weights as weight_trace};
use kv::build as build_kv_cache;
use runtime::{
    backend::BackendInfo,
    trace::{ModelTrace, TraceAction, TraceModel, TraceTokenizer},
};

use super::model::{LoadedExecution, LoadedModel};
use crate::engine::{
    NATIVE_PAGED_ATTENTION_MIN_CONTEXT, paged_attention_enabled, paged_attention_min_context,
};

pub(super) fn build(model: &LoadedModel, backend: BackendInfo) -> ModelTrace {
    let info = &model.info;
    let paged_attention = paged_attention_enabled();
    let paged_attention_min_context =
        paged_attention.then(|| paged_attention_min_context(model.stream()));
    ModelTrace {
        model: TraceModel {
            id: info.manifest.id.clone(),
            root: info.layout.root.display().to_string(),
            model_type: info.metadata.model_type.clone(),
            dtype: info.metadata.dtype.clone(),
            architectures: info.metadata.architectures.clone(),
            context_len: info.metadata.context_len,
            quantization: info.metadata.quantization.clone(),
            quantization_group_size: info.metadata.quantization_group_size,
            quantization_mode: info.metadata.quantization_mode.clone(),
        },
        backend,
        acceleration: vec![
            "safe Rust mirtal execution gateway".into(),
            "explicit GPU stream".into(),
            "MLX Metal graph kernels".into(),
        ],
        decoder: shape::execution(info.decoder.as_ref(), info.encoder.as_ref()),
        tokenizer: tokenizer_trace(model),
        tensors: tensor_trace(model),
        weights: weight_trace(model),
        kv_cache: build_kv_cache(model, paged_attention, paged_attention_min_context),
        actions: actions(model),
        warnings: warnings(model),
    }
}

pub(super) fn emit(trace: &ModelTrace) {
    tracing::debug!(
        model_id = %trace.model.id,
        backend = %trace.backend.name,
        device = %trace.backend.device,
        tensors = trace.tensors.native_tensor_count,
        "native MLX model prepared"
    );
    for action in &trace.actions {
        tracing::debug!(model_id = %trace.model.id, stage = %action.stage, detail = %action.detail, "model trace action");
    }
    for warning in &trace.warnings {
        tracing::warn!(model_id = %trace.model.id, warning, "model trace warning");
    }
}

fn tokenizer_trace(model: &LoadedModel) -> TraceTokenizer {
    model.info.tokenizer.as_ref().map_or_else(
        || TraceTokenizer {
            path: model.info.layout.tokenizer_path.as_ref().map(|path| path.display().to_string()),
            kind: None,
            vocab_size: None,
            stop_token_ids: Vec::new(),
            error: model.info.tokenizer_error.clone(),
        },
        |tokenizer| TraceTokenizer {
            path: Some(tokenizer.path.display().to_string()),
            kind: Some(format!("{:?}", tokenizer.kind)),
            vocab_size: Some(tokenizer.vocab_size),
            stop_token_ids: tokenizer.stop_token_ids.clone(),
            error: model.info.tokenizer_error.clone(),
        },
    )
}

fn actions(model: &LoadedModel) -> Vec<TraceAction> {
    if !matches!(&model.execution, LoadedExecution::Generation(_)) {
        return task_actions(model);
    }
    let info = &model.info;
    let (fused_attention, fused_key_value, fused_gate_up, fused_expert_gate_up) =
        model.fusion_summary();
    let prefix_entries = model.prefix_cache_capacity();
    vec![
        TraceAction::new(
            "inspect",
            format!(
                "compiled semantic execution from {} safetensor shards; tokenizer {}",
                info.layout.weights.len(),
                info.layout.has_tokenizer()
            ),
        ),
        TraceAction::new(
            "load",
            format!(
                "materialized {} MLX safetensors on the CPU load stream; execution uses one GPU stream",
                info.tensor_count
            ),
        ),
        TraceAction::new(
            "linear",
            format!(
                "MLX QMM with fused Q/K/V {fused_attention} layers, K/V {fused_key_value} layers, dense gate/up {fused_gate_up} layers, and expert gate/up {fused_expert_gate_up} layers"
            ),
        ),
        TraceAction::new("expert_fusion", model.expert_fusion_summary()),
        TraceAction::new("moe_router", moe_router_action(model)),
        TraceAction::new("attention", paged_attention_action(model)),
        TraceAction::new("linear_attention", gated_delta_action(model)),
        TraceAction::new(
            "memory",
            format!(
                "Metal wired limit={} bytes, recommended={} bytes; MLX active={} bytes, cache={} bytes, peak={} bytes",
                info.metal_memory.limit,
                info.metal_memory.recommended.unwrap_or(0),
                info.metal_memory.active,
                info.metal_memory.cached,
                info.metal_memory.peak,
            ),
        ),
        TraceAction::new(
            "kv_cache",
            format!(
                "full K/V grows in {}-token allocations; sliding K/V uses a bounded physical ring",
                info.cache_step
            ),
        ),
        TraceAction::new(
            "prefix_cache",
            format!(
                "device-resident K/V and logits snapshots use longest-token-prefix LRU with {prefix_entries} entries"
            ),
        ),
        TraceAction::new(
            "sampling",
            "greedy and bounded top-p/top-k sampling keep token selection plus next decode in MLX; repetition penalty retains the Rust logits fallback",
        ),
        TraceAction::new(
            "prefill",
            format!("causal MLX prefill uses up to {} tokens per graph", info.prefill_step),
        ),
        TraceAction::new("feed_forward", compiled_feed_forward_action()),
    ]
}

fn task_actions(model: &LoadedModel) -> Vec<TraceAction> {
    let execution = match &model.execution {
        LoadedExecution::Embedding(_) => {
            "causal text encoding, last-token pooling, dimension slicing, and L2 normalization"
        },
        LoadedExecution::SequenceScoring(_) => {
            "bidirectional packed-QKV encoding, CLS pooling, and scalar classification"
        },
        LoadedExecution::Generation(_) => unreachable!(),
    };
    vec![
        TraceAction::new(
            "inspect",
            "compiled task semantics from config, task modules, and tensor schema",
        ),
        TraceAction::new(
            "load",
            format!("materialized {} MLX safetensors", model.info.tensor_count),
        ),
        TraceAction::new("execute", execution),
        TraceAction::new("output", "copied only the final task result from Metal"),
    ]
}

fn compiled_feed_forward_action() -> &'static str {
    "SwiGLU and GeGLU use compiled MLX kernels; persistent full feed-forward compilation is disabled"
}

fn moe_router_action(model: &LoadedModel) -> &'static str {
    if model.stream().config().fusion.native_router.enabled() {
        "MoE routing uses the compiled affine router graph"
    } else {
        "MoE routing uses MLX argpartition with selected-logit softmax normalization"
    }
}

fn gated_delta_action(model: &LoadedModel) -> &'static str {
    if model.stream().config().fusion.fused_gated_delta_normalization.enabled() {
        "Gated Delta decode fuses precise Q/K RMS normalization into its recurrent Metal kernel"
    } else {
        "Gated Delta decode uses separate MLX Q/K RMSNorm kernels"
    }
}

fn paged_attention_action(model: &LoadedModel) -> String {
    if model.stream().config().kv_cache.dtype == runtime::kv::KvCacheDType::Int8PerTokenHead {
        return "full-attention K/V uses packed INT8 pages with per-token/head scales and fused dequantization in native paged SDPA".into();
    }
    let minimum = paged_attention_min_context(model.stream());
    format!(
        "full-attention K/V uses canonical head-major pages from {minimum} cached tokens; identity maps expose a zero-copy MLX SDPA view, fragmented COW maps use native paged SDPA immediately, and benchmarked identity shapes switch to native paged SDPA with per-cache scratch at {NATIVE_PAGED_ATTENTION_MIN_CONTEXT} tokens"
    )
}

fn warnings(model: &LoadedModel) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Some(error) = &model.info.tokenizer_error {
        warnings.push(format!("tokenizer report unavailable: {error}"));
    }
    if model.fusion_summary().3 > 0 {
        warnings.push(
            "expert gate/up fusion is enabled by the current Metal-memory policy and trades working-set headroom for decode speed".into(),
        );
    }
    warnings
}
