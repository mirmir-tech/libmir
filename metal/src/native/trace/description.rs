use models::execution::{AttentionFeature, DecoderArchetype};
use runtime::trace::{TraceTensors, TraceWeights};

use crate::native::model::LoadedModel;

pub(super) fn tensors(model: &LoadedModel) -> TraceTensors {
    let info = &model.info;
    TraceTensors {
        tensor_count: info.tensor_count,
        native_tensor_count: info.tensor_count,
        weight_files: info.layout.weights.len(),
        native_shards: info.layout.weights.len(),
        weight_bytes: info.weight_bytes,
        tokenizer: info.layout.has_tokenizer(),
        safetensors_index: info.layout.safetensors_index_path.is_some(),
        readiness: format!("native {:?} execution plan loaded", info.plan.decoder),
        missing: Vec::new(),
        native_dtypes: Vec::new(),
        finite_validation: runtime::trace::TraceFiniteValidation {
            mode: "deferred to MLX kernel execution".into(),
            checked_tensors: 0,
            checked_elements: 0,
        },
    }
}

pub(super) fn weights(model: &LoadedModel) -> TraceWeights {
    let info = &model.info;
    match info.plan.decoder {
        DecoderArchetype::HybridMoe => TraceWeights {
            token_embeddings: "language_model.model.embed_tokens".into(),
            final_norm: "language_model.model.norm.weight".into(),
            output_head: "language_model.model.embed_tokens.weight (tied)".into(),
            output_tied: true,
            layer_count: info.decoder.num_hidden_layers,
            attention_layout: "split Q/K/V; full layers share K/V".into(),
            mlp_layout: "dense GeGLU plus routed quantized MoE".into(),
            linear_bias_count: 0,
        },
        DecoderArchetype::HybridLinearMoe => TraceWeights {
            token_embeddings: "language_model.model.embed_tokens".into(),
            final_norm: "language_model.model.norm.weight".into(),
            output_head: "language_model.lm_head.weight".into(),
            output_tied: false,
            layer_count: info.decoder.num_hidden_layers,
            attention_layout: "Gated Delta recurrence plus gated RMS-normalized GQA".into(),
            mlp_layout: "shared expert routed SwiGLU".into(),
            linear_bias_count: 0,
        },
        DecoderArchetype::DenseSwiGlu => {
            let output_head = if info.decoder.tie_word_embeddings {
                "model.embed_tokens.weight (tied)"
            } else {
                "lm_head.weight"
            };
            TraceWeights {
                token_embeddings: "model.embed_tokens".into(),
                final_norm: "model.norm.weight".into(),
                output_head: output_head.into(),
                output_tied: info.decoder.tie_word_embeddings,
                layer_count: info.decoder.num_hidden_layers,
                attention_layout: dense_attention_layout(info.plan.attention).into(),
                mlp_layout: "dense SwiGLU".into(),
                linear_bias_count: info.decoder.num_hidden_layers.saturating_mul(7),
            }
        },
    }
}

fn dense_attention_layout(feature: AttentionFeature) -> &'static str {
    match feature {
        AttentionFeature::RmsNormalizedGroupedQuery => {
            "split Q/K/V with RMS-normalized grouped-query attention"
        },
        AttentionFeature::GroupedQuery => "split Q/K/V with grouped-query attention",
        AttentionFeature::RmsNormalizedSharedKv
        | AttentionFeature::GatedDeltaAndRmsNormalizedGroupedQuery => unreachable!(),
    }
}
