use models::layout::{AttentionLayerType, DecoderConfig, EncoderConfig};
use runtime::trace::{TraceDecoder, TraceWeights};

use super::{LoadedModel, mlp_layout};

pub(super) fn attention_counts(decoder: &DecoderConfig) -> (usize, usize) {
    decoder.layer_types.iter().fold((0, 0), |(full, sliding), layer| match layer {
        AttentionLayerType::Full => (full + 1, sliding),
        AttentionLayerType::Sliding => (full, sliding + 1),
        AttentionLayerType::Linear => (full, sliding),
    })
}

pub(super) fn execution(
    decoder: Option<&DecoderConfig>,
    encoder: Option<&EncoderConfig>,
    full: usize,
    sliding: usize,
) -> TraceDecoder {
    if let Some(decoder) = decoder {
        return decoder_trace(decoder, full, sliding);
    }
    encoder.map_or_else(empty_execution, encoder_trace)
}

pub(super) fn weights(model: &LoadedModel) -> TraceWeights {
    if let Some(encoder) = model.encoder.as_ref() {
        return TraceWeights {
            token_embeddings: "new.embeddings.word_embeddings".into(),
            final_norm: "post-attention and post-MLP LayerNorm".into(),
            output_head: "pooler tanh and scalar classifier".into(),
            output_tied: false,
            layer_count: encoder.num_hidden_layers,
            attention_layout: "packed QKV with bidirectional F16 attention".into(),
            mlp_layout: mlp_layout(model).into(),
            linear_bias_count: encoder.num_hidden_layers.saturating_mul(4).saturating_add(2),
        };
    }
    let Some(decoder) = model.decoder.as_ref() else {
        return empty_weights();
    };
    TraceWeights {
        token_embeddings: "config-discovered BF16 embedding".into(),
        final_norm: "config-discovered BF16 RMSNorm".into(),
        output_head: "tied or independent checkpoint projection".into(),
        output_tied: decoder.tie_word_embeddings,
        layer_count: decoder.num_hidden_layers,
        attention_layout: "fused QKV with causal paged GQA".into(),
        mlp_layout: mlp_layout(model).into(),
        linear_bias_count: 0,
    }
}

fn encoder_trace(encoder: &EncoderConfig) -> TraceDecoder {
    TraceDecoder {
        layers: encoder.num_hidden_layers,
        hidden_size: encoder.hidden_size,
        intermediate_size: encoder.intermediate_size,
        vocab_size: encoder.vocab_size,
        attention_heads: encoder.num_attention_heads,
        kv_heads: encoder.num_attention_heads,
        head_dim: encoder.head_dim,
        global_head_dim: None,
        global_kv_heads: None,
        full_attention_layers: encoder.num_hidden_layers,
        sliding_attention_layers: 0,
        max_position_embeddings: encoder.max_position_embeddings,
        rope_theta: encoder.rope_theta,
        full_attention_rope_theta: None,
        sliding_attention_rope_theta: None,
        sliding_window: None,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        hidden_activation: Some(encoder.hidden_activation.clone()),
        final_logit_softcapping: None,
    }
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

fn empty_execution() -> TraceDecoder {
    TraceDecoder {
        layers: 0,
        hidden_size: 0,
        intermediate_size: 0,
        vocab_size: 0,
        attention_heads: 0,
        kv_heads: 0,
        head_dim: 0,
        global_head_dim: None,
        global_kv_heads: None,
        full_attention_layers: 0,
        sliding_attention_layers: 0,
        max_position_embeddings: 0,
        rope_theta: None,
        full_attention_rope_theta: None,
        sliding_attention_rope_theta: None,
        sliding_window: None,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        hidden_activation: None,
        final_logit_softcapping: None,
    }
}

fn empty_weights() -> TraceWeights {
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
