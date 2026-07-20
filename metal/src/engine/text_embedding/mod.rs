mod attention;
mod layer;
mod model;
mod weights;

pub use model::TextEmbeddingModel;
use models::layout::DecoderConfig;

use crate::engine::{Error, Result};

#[derive(Debug, Clone, Copy)]
struct LayerConfig {
    index: usize,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    query_width: i32,
    attention_scale: f32,
    rope_base: f32,
    rms_norm_eps: f32,
}

impl LayerConfig {
    fn from_decoder(index: usize, decoder: &DecoderConfig) -> Result<Self> {
        if decoder.num_experts.is_some()
            || decoder.sliding_window.is_some()
            || decoder.hidden_activation.as_deref() != Some("silu")
        {
            return Err(Error::InvalidModel(
                "text embedding decoder requires dense full-attention SwiGLU".into(),
            ));
        }
        if decoder.rope_scaling.is_some() {
            return Err(Error::InvalidModel(
                "text embedding decoder does not yet support scaled RoPE".into(),
            ));
        }
        let head_dim = i32::try_from(decoder.head_dim)?;
        let heads = i32::try_from(decoder.num_attention_heads)?;
        Ok(Self {
            index,
            heads,
            kv_heads: i32::try_from(decoder.num_key_value_heads)?,
            head_dim,
            query_width: heads.checked_mul(head_dim).ok_or(Error::ShapeOverflow)?,
            attention_scale: head_dim.to_string().parse::<f32>()?.sqrt().recip(),
            rope_base: decoder.rope_theta.unwrap_or(10_000.0).to_string().parse()?,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_width_is_independent_from_hidden_size() -> Result<()> {
        let decoder = DecoderConfig::from_value(&serde_json::json!({
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151_936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 0.000_001,
            "hidden_act": "silu"
        }))?;

        let config = LayerConfig::from_decoder(0, &decoder)?;

        assert_eq!(config.query_width, 2048);
        assert_ne!(config.query_width, i32::try_from(decoder.hidden_size)?);
        Ok(())
    }
}
