use serde_json::Value;

use super::{AttentionOutput, DecoderConfig, parse::invalid};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayerType {
    Full,
    Linear,
    Sliding,
}

pub(super) fn layer_types(value: &Value, layer_count: usize) -> Result<Vec<AttentionLayerType>> {
    let Some(raw) = value.get("layer_types") else {
        return Ok(vec![AttentionLayerType::Full; layer_count]);
    };
    let Some(items) = raw.as_array() else {
        return Err(invalid("layer_types must be an array"));
    };
    if items.len() != layer_count {
        return Err(invalid(format!(
            "layer_types has {} entries, expected {layer_count}",
            items.len()
        )));
    }
    items.iter().map(layer_type).collect()
}

pub(super) fn rope_theta(value: &Value, layer_type: &str) -> Option<f64> {
    value
        .get("rope_parameters")
        .and_then(|parameters| parameters.get(layer_type))
        .and_then(|typed| typed.get("rope_theta"))
        .and_then(Value::as_f64)
}

pub(super) fn rope_type(value: &Value, layer_type: &str) -> Result<Option<String>> {
    let Some(raw) = rope_field(value, layer_type, "rope_type") else {
        return Ok(None);
    };
    raw.as_str()
        .map(|text| Some(text.to_owned()))
        .ok_or_else(|| invalid("rope_type must be a string"))
}

pub(super) fn partial_rotary_factor(value: &Value, layer_type: &str) -> Result<Option<f64>> {
    let Some(raw) = rope_field(value, layer_type, "partial_rotary_factor") else {
        return Ok(None);
    };
    raw.as_f64()
        .map(Some)
        .ok_or_else(|| invalid("partial_rotary_factor must be a number"))
}

fn rope_field<'a>(value: &'a Value, layer_type: &str, field: &str) -> Option<&'a Value> {
    value.get("rope_parameters")?.get(layer_type)?.get(field)
}

fn layer_type(value: &Value) -> Result<AttentionLayerType> {
    match value.as_str() {
        Some("full_attention") => Ok(AttentionLayerType::Full),
        Some("linear_attention") => Ok(AttentionLayerType::Linear),
        Some("sliding_attention") => Ok(AttentionLayerType::Sliding),
        Some(other) => Err(invalid(format!("unsupported layer_type {other}"))),
        None => Err(invalid("layer_types entries must be strings")),
    }
}

impl DecoderConfig {
    #[must_use]
    pub fn uses_hybrid_routed_moe_stack(&self) -> bool {
        self.attention_k_eq_v
            && self.num_experts.is_some()
            && self.hidden_activation.as_deref() == Some("gelu_pytorch_tanh")
    }

    #[must_use]
    pub fn uses_hybrid_linear_moe_stack(&self) -> bool {
        self.linear_attention.is_some()
            && self.num_experts.is_some()
            && self.shared_expert_intermediate_size.is_some()
            && self.attention_output == AttentionOutput::Gated
            && self.layer_types.contains(&AttentionLayerType::Linear)
            && self.layer_types.contains(&AttentionLayerType::Full)
    }

    #[must_use]
    pub fn layer_type(&self, index: usize) -> AttentionLayerType {
        self.layer_types.get(index).copied().unwrap_or(AttentionLayerType::Full)
    }

    #[must_use]
    pub fn layer_sliding_window(&self, index: usize) -> Option<usize> {
        self.sliding_window
            .filter(|_| self.layer_type(index) == AttentionLayerType::Sliding)
    }

    #[must_use]
    pub fn layer_head_dim(&self, index: usize) -> usize {
        match self.layer_type(index) {
            AttentionLayerType::Full => self.global_head_dim.unwrap_or(self.head_dim),
            AttentionLayerType::Linear | AttentionLayerType::Sliding => self.head_dim,
        }
    }

    #[must_use]
    pub fn layer_key_value_heads(&self, index: usize) -> usize {
        if self.layer_type(index) == AttentionLayerType::Full && self.attention_k_eq_v {
            self.num_global_key_value_heads.unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    #[must_use]
    pub fn rope_theta_for_layer(&self, index: usize) -> Option<f64> {
        match self.layer_type(index) {
            AttentionLayerType::Full => self.full_attention_rope_theta.or(self.rope_theta),
            AttentionLayerType::Linear => None,
            AttentionLayerType::Sliding => self.sliding_attention_rope_theta.or(self.rope_theta),
        }
    }

    #[must_use]
    pub fn rope_type_for_layer(&self, index: usize) -> Option<&str> {
        match self.layer_type(index) {
            AttentionLayerType::Full => self.full_attention_rope_type.as_deref(),
            AttentionLayerType::Linear => None,
            AttentionLayerType::Sliding => self.sliding_attention_rope_type.as_deref(),
        }
    }

    #[must_use]
    pub fn partial_rotary_factor_for_layer(&self, index: usize) -> Option<f64> {
        match self.layer_type(index) {
            AttentionLayerType::Full => self.full_attention_partial_rotary_factor,
            AttentionLayerType::Linear => None,
            AttentionLayerType::Sliding => self.sliding_attention_partial_rotary_factor,
        }
    }
}
