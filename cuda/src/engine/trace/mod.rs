use std::collections::BTreeMap;

use models::{
    layout::{AttentionLayerType, DecoderConfig},
    tokenizer::TextTokenizer,
};
use runtime::{
    backend::BackendInfo,
    kv::CacheConfig,
    trace::{
        ModelTrace, TraceDTypeCount, TraceDecoder, TraceFiniteValidation, TraceKvCache, TraceModel,
        TraceTensors, TraceTokenizer, TraceWeights,
    },
};

use super::model::LoadedModel;
use crate::Result;

mod execution;
use execution::{acceleration, actions, mlp_layout, warnings};

pub(super) fn build(
    model: &LoadedModel,
    backend: BackendInfo,
    cache: CacheConfig,
) -> Result<ModelTrace> {
    let (full_attention_layers, sliding_attention_layers) = attention_counts(&model.decoder);
    let sessions = model.sessions()?.len();
    Ok(ModelTrace {
        model: TraceModel {
            id: model.manifest.id.clone(),
            root: model.layout.root.display().to_string(),
            family: model.metadata.family.clone(),
            model_type: model.metadata.model_type.clone(),
            dtype: model.metadata.dtype.clone(),
            architectures: model.metadata.architectures.clone(),
            context_len: model.metadata.context_len,
            quantization: model.metadata.quantization.clone(),
            quantization_group_size: model.metadata.quantization_group_size,
            quantization_mode: model.metadata.quantization_mode.clone(),
        },
        backend,
        acceleration: acceleration(model),
        decoder: decoder_trace(&model.decoder, full_attention_layers, sliding_attention_layers),
        tokenizer: tokenizer_trace(model),
        tensors: tensor_trace(model),
        weights: TraceWeights {
            token_embeddings: "config-discovered BF16 embedding".into(),
            final_norm: "config-discovered BF16 RMSNorm".into(),
            output_head: "tied or independent checkpoint projection".into(),
            output_tied: model.decoder.tie_word_embeddings,
            layer_count: model.decoder.num_hidden_layers,
            attention_layout: "fused QKV with causal paged GQA".into(),
            mlp_layout: mlp_layout(model).into(),
            linear_bias_count: 0,
        },
        kv_cache: TraceKvCache {
            dtype: cache.dtype,
            quant_mode: cache.dtype.quant_mode(),
            scale_granularity: cache.dtype.scale_granularity(),
            decode_attention: "native split-KV paged CUDA attention".into(),
            block_size: Some(cache.block_size),
            physical_page_key: "runtime BlockId".into(),
            prefix_cache: true,
            paged_attention: true,
            paged_attention_min_context: Some(1),
            entry_count: sessions,
            cached_tokens: 0,
            resident_token_slots: cache.block_size * cache.block_count as usize,
        },
        actions: actions(model, cache),
        warnings: warnings(model),
    })
}

fn decoder_trace(decoder: &DecoderConfig, full: usize, sliding: usize) -> TraceDecoder {
    TraceDecoder {
        layers: decoder.num_hidden_layers,
        hidden_size: decoder.hidden_size,
        intermediate_size: decoder.intermediate_size,
        vocab_size: decoder.vocab_size,
        attention_heads: decoder.num_attention_heads,
        kv_heads: decoder.num_key_value_heads,
        head_dim: decoder.head_dim,
        global_head_dim: decoder.global_head_dim,
        global_kv_heads: decoder.num_global_key_value_heads,
        full_attention_layers: full,
        sliding_attention_layers: sliding,
        max_position_embeddings: decoder.max_position_embeddings,
        rope_theta: decoder.rope_theta,
        full_attention_rope_theta: decoder.full_attention_rope_theta,
        sliding_attention_rope_theta: decoder.sliding_attention_rope_theta,
        sliding_window: decoder.sliding_window,
        num_experts: decoder.num_experts,
        top_k_experts: decoder.top_k_experts,
        moe_intermediate_size: decoder.moe_intermediate_size,
        hidden_activation: decoder.hidden_activation.clone(),
        final_logit_softcapping: decoder.final_logit_softcapping,
    }
}

fn tokenizer_trace(model: &LoadedModel) -> TraceTokenizer {
    match TextTokenizer::from_layout(&model.layout) {
        Ok(tokenizer) => {
            let info = tokenizer.info();
            TraceTokenizer {
                path: Some(info.path.display().to_string()),
                kind: Some(format!("{:?}", info.kind)),
                vocab_size: Some(info.vocab_size),
                stop_token_ids: info.stop_token_ids,
                error: None,
            }
        },
        Err(error) => TraceTokenizer {
            path: model.layout.tokenizer_path.as_ref().map(|path| path.display().to_string()),
            kind: None,
            vocab_size: None,
            stop_token_ids: Vec::new(),
            error: Some(error.to_string()),
        },
    }
}

fn tensor_trace(model: &LoadedModel) -> TraceTensors {
    let mut dtypes = BTreeMap::<String, (usize, usize)>::new();
    for tensor in &model.catalog.tensors {
        let entry = dtypes.entry(tensor.dtype.clone()).or_default();
        entry.0 += 1;
        entry.1 += tensor.shape.iter().product::<usize>();
    }
    let readiness = model.vision_readiness.as_ref().map_or_else(
        || format!("native CUDA {:?} model loaded", model.plan.decoder),
        |vision| {
            format!(
                "native CUDA {:?} text model loaded; {:?} vision discovered; {}",
                model.plan.decoder,
                model.vision.as_ref().map(models::layout::VisionConfig::pipeline),
                vision.summary()
            )
        },
    );
    TraceTensors {
        tensor_count: model.catalog.len(),
        native_tensor_count: model.catalog.len(),
        weight_files: model.layout.weights.len(),
        native_shards: model.layout.weights.len(),
        weight_bytes: model.layout.weights.iter().map(|weight| weight.bytes).sum(),
        tokenizer: model.layout.has_tokenizer(),
        safetensors_index: model.layout.safetensors_index_path.is_some(),
        readiness,
        missing: model
            .vision_readiness
            .as_ref()
            .map(|vision| vision.missing.clone())
            .unwrap_or_default(),
        native_dtypes: dtypes
            .into_iter()
            .map(|(dtype, (tensors, elements))| TraceDTypeCount { dtype, tensors, elements })
            .collect(),
        finite_validation: TraceFiniteValidation {
            mode: "checkpoint metadata".into(),
            checked_tensors: 0,
            checked_elements: 0,
        },
    }
}

fn attention_counts(decoder: &DecoderConfig) -> (usize, usize) {
    decoder.layer_types.iter().fold((0, 0), |(full, sliding), layer| match layer {
        AttentionLayerType::Full => (full + 1, sliding),
        AttentionLayerType::Sliding => (full, sliding + 1),
        AttentionLayerType::Linear => (full, sliding),
    })
}
