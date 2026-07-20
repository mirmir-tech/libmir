use models::layout::{AttentionLayerType, DecoderConfig, EncoderConfig};
use runtime::trace::TraceDecoder;

pub(super) fn execution(
    decoder: Option<&DecoderConfig>,
    encoder: Option<&EncoderConfig>,
) -> TraceDecoder {
    if let Some(decoder) = decoder {
        let (full_attention_layers, sliding_attention_layers) = attention_counts(decoder);
        return TraceDecoder {
            layers: decoder.num_hidden_layers,
            hidden_size: decoder.hidden_size,
            intermediate_size: decoder.intermediate_size,
            vocab_size: decoder.vocab_size,
            attention_heads: decoder.num_attention_heads,
            kv_heads: decoder.num_key_value_heads,
            head_dim: decoder.head_dim,
            global_head_dim: decoder.global_head_dim,
            global_kv_heads: decoder.num_global_key_value_heads,
            full_attention_layers,
            sliding_attention_layers,
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
        };
    }
    encoder.map_or_else(empty_shape, encoder_shape)
}

fn empty_shape() -> TraceDecoder {
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

fn encoder_shape(encoder: &EncoderConfig) -> TraceDecoder {
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

fn attention_counts(decoder: &DecoderConfig) -> (usize, usize) {
    decoder.layer_types.iter().fold((0, 0), |(full, sliding), layer| match layer {
        AttentionLayerType::Full => (full + 1, sliding),
        AttentionLayerType::Linear => (full, sliding),
        AttentionLayerType::Sliding => (full, sliding + 1),
    })
}
