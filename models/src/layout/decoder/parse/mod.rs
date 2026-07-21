use std::{fs, path::Path};

use serde_json::Value;

use super::{DecoderConfig, attention, features, rope};
use crate::{error::Result, layout::ModelLayout};

mod value;

pub(super) use value::{invalid, optional_usize, require_usize};
use value::{optional_bool, optional_f64, optional_string, partial_rotary_factor, rope_theta};

impl DecoderConfig {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        Self::from_config_path(&layout.config_path)
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&json)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let decoder = decoder_value(value);
        let num_attention_heads = require_usize(decoder, &["num_attention_heads", "n_head"])?;
        if num_attention_heads == 0 {
            return Err(invalid("num_attention_heads must be greater than zero"));
        }
        let hidden_size = require_usize(decoder, &["hidden_size", "n_embd"])?;
        let num_hidden_layers =
            require_usize(decoder, &["num_hidden_layers", "num_layers", "n_layer"])?;
        let partial_rotary_factor = partial_rotary_factor(decoder)?;
        let explicit_head_dim =
            optional_usize(decoder, &["head_dim", "attention_head_dim", "kv_channels"])?;
        if explicit_head_dim.is_none() && !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(invalid("hidden_size must be divisible by num_attention_heads"));
        }
        let head_dim = explicit_head_dim.unwrap_or(hidden_size / num_attention_heads);
        let config = Self {
            hidden_size,
            intermediate_size: intermediate_size(decoder)?,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: optional_usize(
                decoder,
                &["num_key_value_heads", "multi_query_group_num", "n_kv_heads"],
            )?
            .unwrap_or(num_attention_heads),
            head_dim,
            global_head_dim: optional_usize(decoder, &["global_head_dim"])?,
            num_global_key_value_heads: optional_usize(decoder, &["num_global_key_value_heads"])?,
            vocab_size: require_usize(decoder, &["vocab_size"])?,
            max_position_embeddings: optional_usize(
                decoder,
                &["max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"],
            )?
            .unwrap_or(4096),
            rms_norm_eps: optional_f64(
                decoder,
                &["rms_norm_eps", "layer_norm_epsilon", "layernorm_epsilon", "norm_epsilon"],
            )?
            .unwrap_or(1e-5),
            rope_theta: rope_theta(decoder)?,
            rope_scaling: rope::scaling(decoder)?,
            partial_rotary_factor,
            rope_layout: features::rope_layout(decoder)?,
            full_attention_rope_theta: attention::rope_theta(decoder, "full_attention"),
            sliding_attention_rope_theta: attention::rope_theta(decoder, "sliding_attention"),
            full_attention_rope_type: attention::rope_type(decoder, "full_attention")?,
            sliding_attention_rope_type: attention::rope_type(decoder, "sliding_attention")?,
            full_attention_partial_rotary_factor: attention::partial_rotary_factor(
                decoder, "full_attention",
            )?
            .or(partial_rotary_factor),
            sliding_attention_partial_rotary_factor: attention::partial_rotary_factor(
                decoder,
                "sliding_attention",
            )?
            .or(partial_rotary_factor),
            layer_types: attention::layer_types(decoder, num_hidden_layers)?,
            tie_word_embeddings: optional_bool(decoder, "tie_word_embeddings")?
                .or(optional_bool(value, "tie_word_embeddings")?)
                .unwrap_or(false),
            attention_k_eq_v: optional_bool(decoder, "attention_k_eq_v")?.unwrap_or(false),
            attention_scale: attention_scale(decoder)?,
            attention_output: features::attention_output(decoder)?,
            sliding_window: optional_usize(decoder, &["sliding_window"])?,
            linear_attention: features::linear_attention(decoder)?,
            num_experts: optional_usize(decoder, &["num_experts", "num_local_experts"])?,
            top_k_experts: optional_usize(
                decoder,
                &["top_k_experts", "num_experts_per_tok", "experts_per_token"],
            )?,
            moe_intermediate_size: optional_usize(decoder, &["moe_intermediate_size"])?,
            shared_expert_intermediate_size: optional_usize(
                decoder,
                &["shared_expert_intermediate_size"],
            )?,
            hidden_activation: optional_string(decoder, "hidden_activation")?
                .or(optional_string(decoder, "hidden_act")?),
            final_logit_softcapping: optional_f64(decoder, &["final_logit_softcapping"])?,
            attention_bias: optional_bool(decoder, "attention_bias")?.unwrap_or(false),
            attention_sinks: optional_bool(decoder, "attention_sinks")?.unwrap_or(false),
            swiglu_limit: optional_f64(decoder, &["swiglu_limit"])?,
            initial_context_length: optional_usize(decoder, &["initial_context_length"])?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.num_key_value_heads == 0 || self.head_dim == 0 {
            return Err(invalid("KV head count and head_dim must be greater than zero"));
        }
        if self.global_head_dim == Some(0) || self.num_global_key_value_heads == Some(0) {
            return Err(invalid("global attention dimensions must be greater than zero"));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(invalid("num_attention_heads must be divisible by num_key_value_heads"));
        }
        if self.sliding_window == Some(0) {
            return Err(invalid("sliding_window must be greater than zero"));
        }
        if self.attention_scale.is_some_and(|scale| !scale.is_finite() || scale <= 0.0) {
            return Err(invalid("attention_scale must be finite and greater than zero"));
        }
        if self.num_experts == Some(0) || self.top_k_experts == Some(0) {
            return Err(invalid("MoE expert counts must be greater than zero"));
        }
        if let (Some(top_k), Some(experts)) = (self.top_k_experts, self.num_experts)
            && top_k > experts
        {
            return Err(invalid("top_k_experts cannot exceed num_experts"));
        }
        Ok(())
    }
}

fn intermediate_size(value: &Value) -> Result<usize> {
    optional_usize(
        value,
        &[
            "intermediate_size",
            "ffn_hidden_size",
            "n_inner",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ],
    )?
    .ok_or_else(|| {
        invalid("missing intermediate_size or ffn_hidden_size or n_inner or MoE intermediate size")
    })
}

fn decoder_value(value: &Value) -> &Value {
    ["text_config", "language_config"]
        .into_iter()
        .filter_map(|key| value.get(key))
        .find(|section| section.is_object())
        .unwrap_or(value)
}

fn attention_scale(decoder: &Value) -> Result<Option<f64>> {
    optional_f64(decoder, &["attention_scale", "attention_multiplier"])
}
