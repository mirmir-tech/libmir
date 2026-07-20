use std::{fs, path::Path};

use serde_json::Value;

use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncoderRopeScaling {
    Ntk { factor: f64, mixed_b: Option<f64> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderPositionEmbedding {
    Absolute,
    Rope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    LayerNorm,
    RmsNorm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub hidden_activation: String,
    pub position_embedding: EncoderPositionEmbedding,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<EncoderRopeScaling>,
    pub norm: NormKind,
    pub type_vocab_size: usize,
    pub packed_qkv: bool,
    pub num_labels: usize,
}

impl EncoderConfig {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let json = fs::read_to_string(&layout.config_path)?;
        Self::from_value(&serde_json::from_str(&json)?)
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        Self::from_value(&serde_json::from_str(&json)?)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        let hidden_size = usize_field(value, "hidden_size")?;
        let heads = usize_field(value, "num_attention_heads")?;
        if heads == 0 || hidden_size == 0 || !hidden_size.is_multiple_of(heads) {
            return Err(invalid("encoder hidden_size must be divisible by a non-zero head count"));
        }
        let position_embedding = match string_field(value, "position_embedding_type")?.as_str() {
            "absolute" => EncoderPositionEmbedding::Absolute,
            "rope" => EncoderPositionEmbedding::Rope,
            other => {
                return Err(invalid(format!("unsupported encoder position embedding {other}")));
            },
        };
        let norm = match string_field(value, "layer_norm_type")?.as_str() {
            "layer_norm" => NormKind::LayerNorm,
            "rms_norm" => NormKind::RmsNorm,
            other => return Err(invalid(format!("unsupported encoder norm {other}"))),
        };
        Ok(Self {
            hidden_size,
            intermediate_size: usize_field(value, "intermediate_size")?,
            num_hidden_layers: usize_field(value, "num_hidden_layers")?,
            num_attention_heads: heads,
            head_dim: hidden_size / heads,
            vocab_size: usize_field(value, "vocab_size")?,
            max_position_embeddings: usize_field(value, "max_position_embeddings")?,
            layer_norm_eps: number_field(value, "layer_norm_eps")?,
            hidden_activation: string_field(value, "hidden_act")?,
            position_embedding,
            rope_theta: optional_number(value, "rope_theta"),
            rope_scaling: rope_scaling(value)?,
            norm,
            type_vocab_size: optional_usize(value, "type_vocab_size")?.unwrap_or(0),
            packed_qkv: value.get("pack_qkv").and_then(Value::as_bool).unwrap_or(false),
            num_labels: optional_usize(value, "num_labels")?.unwrap_or(0),
        })
    }
}

fn rope_scaling(value: &Value) -> Result<Option<EncoderRopeScaling>> {
    let Some(scaling) = value.get("rope_scaling").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let kind = scaling
        .get("type")
        .or_else(|| scaling.get("rope_type"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("encoder rope_scaling requires type"))?;
    if kind != "ntk" {
        return Err(invalid(format!("unsupported encoder rope scaling {kind}")));
    }
    let factor = number_field(scaling, "factor")?;
    if !factor.is_finite() || factor <= 0.0 {
        return Err(invalid("encoder NTK factor must be finite and positive"));
    }
    Ok(Some(EncoderRopeScaling::Ntk {
        factor,
        mixed_b: optional_number(scaling, "mixed_b"),
    }))
}

fn usize_field(value: &Value, field: &str) -> Result<usize> {
    optional_usize(value, field)?.ok_or_else(|| invalid(format!("missing encoder {field}")))
}

fn optional_usize(value: &Value, field: &str) -> Result<Option<usize>> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|value| Ok(usize::try_from(value)?))
        .transpose()
}

fn number_field(value: &Value, field: &str) -> Result<f64> {
    optional_number(value, field).ok_or_else(|| invalid(format!("missing encoder {field}")))
}

fn optional_number(value: &Value, field: &str) -> Option<f64> {
    value.get(field).and_then(Value::as_f64)
}

fn string_field(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("missing encoder {field}")))
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn reads_packed_post_norm_ntk_encoder_without_model_identity() -> Result<()> {
        let config = EncoderConfig::from_value(&json!({
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 250_002,
            "max_position_embeddings": 8192,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
            "position_embedding_type": "rope",
            "rope_theta": 20000.0,
            "rope_scaling": {"type": "ntk", "factor": 8.0},
            "layer_norm_type": "layer_norm",
            "type_vocab_size": 1,
            "pack_qkv": true,
            "num_labels": 1
        }))?;

        assert_eq!(config.head_dim, 64);
        assert_eq!(
            config.rope_scaling,
            Some(EncoderRopeScaling::Ntk { factor: 8.0, mixed_b: None })
        );
        assert!(config.packed_qkv);
        assert_eq!(config.num_labels, 1);
        Ok(())
    }
}
