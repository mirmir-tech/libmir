use models::execution::{AttentionFeature, DecoderArchetype};
use runtime::trace::{TraceTensors, TraceWeights};

use crate::native::model::LoadedModel;

pub(super) fn tensors(model: &LoadedModel) -> TraceTensors {
    let info = &model.info;
    let readiness = info.vision_readiness.as_ref().map_or_else(
        || format!("native {:?} task execution plan loaded", info.task_plan.task()),
        |vision| {
            format!(
                "native {:?} text execution plan loaded; {:?} discovered; {}",
                info.task_plan.task(),
                info.vision.as_ref().map(models::layout::VisionConfig::pipeline),
                vision.summary()
            )
        },
    );
    TraceTensors {
        tensor_count: info.tensor_count,
        native_tensor_count: info.tensor_count,
        weight_files: info.layout.weights.len(),
        native_shards: info.layout.weights.len(),
        weight_bytes: info.weight_bytes,
        tokenizer: info.layout.has_tokenizer(),
        safetensors_index: info.layout.safetensors_index_path.is_some(),
        readiness,
        missing: info
            .vision_readiness
            .as_ref()
            .map(|vision| vision.missing.clone())
            .unwrap_or_default(),
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
    let Some(plan) = info.plan.as_ref() else {
        let Some(encoder) = info.encoder.as_ref() else {
            return unknown_weights();
        };
        return TraceWeights {
            token_embeddings: "new.embeddings.word_embeddings".into(),
            final_norm: "post-attention and post-MLP LayerNorm per encoder layer".into(),
            output_head: "new.pooler.dense -> tanh -> classifier".into(),
            output_tied: false,
            layer_count: encoder.num_hidden_layers,
            attention_layout: "packed QKV with bidirectional self-attention".into(),
            mlp_layout: "packed gated exact-GELU feed-forward".into(),
            linear_bias_count: encoder.num_hidden_layers.saturating_mul(4).saturating_add(2),
        };
    };
    let Some(decoder) = info.decoder.as_ref() else {
        return unknown_weights();
    };
    match plan.decoder {
        DecoderArchetype::HybridMoe => TraceWeights {
            token_embeddings: "language_model.model.embed_tokens".into(),
            final_norm: "language_model.model.norm.weight".into(),
            output_head: "language_model.model.embed_tokens.weight (tied)".into(),
            output_tied: true,
            layer_count: decoder.num_hidden_layers,
            attention_layout: "split Q/K/V; full layers share K/V".into(),
            mlp_layout: "dense GeGLU plus routed quantized MoE".into(),
            linear_bias_count: 0,
        },
        DecoderArchetype::HybridLinearMoe => TraceWeights {
            token_embeddings: "language_model.model.embed_tokens".into(),
            final_norm: "language_model.model.norm.weight".into(),
            output_head: "language_model.lm_head.weight".into(),
            output_tied: false,
            layer_count: decoder.num_hidden_layers,
            attention_layout: "Gated Delta recurrence plus gated RMS-normalized GQA".into(),
            mlp_layout: "shared expert routed SwiGLU".into(),
            linear_bias_count: 0,
        },
        DecoderArchetype::DenseSwiGlu => {
            let output_head = if decoder.tie_word_embeddings {
                "model.embed_tokens.weight (tied)"
            } else {
                "lm_head.weight"
            };
            TraceWeights {
                token_embeddings: "model.embed_tokens".into(),
                final_norm: "model.norm.weight".into(),
                output_head: output_head.into(),
                output_tied: decoder.tie_word_embeddings,
                layer_count: decoder.num_hidden_layers,
                attention_layout: dense_attention_layout(plan.attention).into(),
                mlp_layout: "dense SwiGLU".into(),
                linear_bias_count: decoder.num_hidden_layers.saturating_mul(7),
            }
        },
    }
}

fn unknown_weights() -> TraceWeights {
    TraceWeights {
        token_embeddings: "unknown".into(),
        final_norm: "unknown".into(),
        output_head: "unknown".into(),
        output_tied: false,
        layer_count: 0,
        attention_layout: "unknown".into(),
        mlp_layout: "unknown".into(),
        linear_bias_count: 0,
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
