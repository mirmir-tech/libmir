use std::collections::BTreeMap;

use serde::Deserialize;

use super::{ModernBertAttention, ModernBertConfig, invalid};
use crate::error::{ModelsError, Result};

/// `config.json` as written by transformers for `model_type: modernbert`.
#[derive(Debug, Deserialize)]
pub(super) struct RawConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    norm_eps: f64,
    hidden_activation: String,
    local_attention: usize,
    #[serde(default)]
    layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    global_attn_every_n_layers: Option<usize>,
    #[serde(default)]
    rope_parameters: Option<BTreeMap<LayerType, RopeParameters>>,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    attention_bias: bool,
    mlp_bias: bool,
    norm_bias: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    rope_type: String,
}

impl TryFrom<RawConfig> for ModernBertConfig {
    type Error = ModelsError;

    fn try_from(raw: RawConfig) -> Result<Self> {
        if raw.model_type != "modernbert" {
            return Err(invalid(format!("expected modernbert, found {}", raw.model_type)));
        }
        if raw.attention_bias || raw.mlp_bias || raw.norm_bias {
            return Err(invalid("ModernBERT with biases is not supported"));
        }
        if raw.hidden_activation != "gelu" {
            return Err(invalid(format!(
                "ModernBERT activation {} is not supported",
                raw.hidden_activation
            )));
        }
        let heads = raw.num_attention_heads;
        if heads == 0 || !raw.hidden_size.is_multiple_of(heads) {
            return Err(invalid("hidden_size must be divisible by a non-zero head count"));
        }
        let radius = raw.local_attention / 2;
        let layers = raw
            .layer_types()?
            .into_iter()
            .map(|layer| match layer {
                LayerType::FullAttention => ModernBertAttention::Global,
                LayerType::SlidingAttention => ModernBertAttention::Local { radius },
            })
            .collect();
        let (global_rope_theta, local_rope_theta) = raw.rope_thetas()?;
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_attention_heads: heads,
            head_dim: raw.hidden_size / heads,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            norm_eps: raw.norm_eps,
            layers,
            global_rope_theta,
            local_rope_theta,
        })
    }
}

impl RawConfig {
    fn layer_types(&self) -> Result<Vec<LayerType>> {
        let layers = match (&self.layer_types, self.global_attn_every_n_layers) {
            (Some(layers), _) => layers.clone(),
            (None, Some(every)) if every > 0 => (0..self.num_hidden_layers)
                .map(|index| {
                    if index.is_multiple_of(every) {
                        LayerType::FullAttention
                    } else {
                        LayerType::SlidingAttention
                    }
                })
                .collect(),
            (None, _) => {
                return Err(invalid("ModernBERT needs layer_types or global_attn_every_n_layers"));
            },
        };
        if layers.len() != self.num_hidden_layers {
            return Err(invalid(format!(
                "ModernBERT lists {} layer types for {} layers",
                layers.len(),
                self.num_hidden_layers
            )));
        }
        Ok(layers)
    }

    fn rope_thetas(&self) -> Result<(f64, f64)> {
        let (global, local) = match &self.rope_parameters {
            Some(parameters) => (
                default_rope(parameters, LayerType::FullAttention)?,
                default_rope(parameters, LayerType::SlidingAttention)?,
            ),
            None => self.global_rope_theta.zip(self.local_rope_theta).ok_or_else(|| {
                invalid("ModernBERT needs rope_parameters or both global and local rope thetas")
            })?,
        };
        if !(global.is_finite() && global > 0.0 && local.is_finite() && local > 0.0) {
            return Err(invalid("ModernBERT rope thetas must be finite and positive"));
        }
        Ok((global, local))
    }
}

fn default_rope(parameters: &BTreeMap<LayerType, RopeParameters>, layer: LayerType) -> Result<f64> {
    let rope = parameters
        .get(&layer)
        .ok_or_else(|| invalid(format!("ModernBERT has no rope parameters for {layer:?}")))?;
    if rope.rope_type != "default" {
        return Err(invalid(format!("ModernBERT rope type {} is not supported", rope.rope_type)));
    }
    Ok(rope.rope_theta)
}
