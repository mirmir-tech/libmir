//! Per-layer attention traits beyond the layer type: layers that attend
//! without a position encoding, chunked attention on the rotary ones, a
//! query temperature on the others, and unit Q/K norms (Cohere 2, Llama 4).

use serde_json::Value;

use super::{
    AttentionLayerType,
    defaults::{AttentionPatternDefault, Defaults},
    parse::{invalid, optional_bool, optional_f64, optional_usize},
};
use crate::{error::Result, semantic::QueryTemperature};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttentionPattern {
    /// Layers that attend without a position encoding; empty when every
    /// layer rotates.
    pub rotation_free: Vec<bool>,
    /// Rotary layers attend within `chunk`-aligned blocks (Llama 4's
    /// `attention_chunk_size`).
    pub rotary_chunk: Option<usize>,
    /// Scale on the queries of rotation-free layers (Llama 4's
    /// `attn_temperature_tuning`).
    pub rotation_free_temperature: Option<QueryTemperature>,
    /// Rotary layers normalize each query and key head without a learned
    /// scale (Llama 4's `use_qk_norm`).
    pub rotary_unit_qk_norm: bool,
}

impl AttentionPattern {
    /// Whether layer `index` rotates its queries and keys.
    #[must_use]
    pub fn rotates(&self, index: usize) -> bool {
        !self.rotation_free.get(index).copied().unwrap_or(false)
    }
}

pub(super) fn parse(
    decoder: &Value,
    defaults: &Defaults,
    layer_types: &[AttentionLayerType],
) -> Result<AttentionPattern> {
    match defaults.attention_pattern {
        AttentionPatternDefault::Uniform => Ok(AttentionPattern::default()),
        // transformers `Cohere2Attention`: only sliding layers rotate.
        AttentionPatternDefault::SlidingRotary => Ok(AttentionPattern {
            rotation_free: layer_types
                .iter()
                .map(|kind| *kind != AttentionLayerType::Sliding)
                .collect(),
            ..AttentionPattern::default()
        }),
        AttentionPatternDefault::Llama4 => llama4(decoder, layer_types.len()),
    }
}

/// transformers `Llama4TextConfig`: `no_rope_layers[i]` is one where layer
/// `i` rotates, every fourth layer by default; chunked attention, the
/// temperature and the unit Q/K norms follow.
fn llama4(decoder: &Value, layers: usize) -> Result<AttentionPattern> {
    let rotation_free = match decoder.get("no_rope_layers").and_then(Value::as_array) {
        Some(flags) if !flags.is_empty() => {
            if flags.len() != layers {
                return Err(invalid("no_rope_layers must name every layer"));
            }
            flags.iter().map(|flag| flag.as_u64() == Some(0)).collect()
        },
        _ => {
            let interval = optional_usize(decoder, &["no_rope_layer_interval"])?.unwrap_or(4);
            if interval == 0 {
                return Err(invalid("no_rope_layer_interval must be positive"));
            }
            (0..layers).map(|index| (index + 1) % interval == 0).collect()
        },
    };
    let temperature = optional_bool(decoder, "attn_temperature_tuning")?
        .unwrap_or(true)
        .then(|| {
            Ok::<_, crate::error::ModelsError>(QueryTemperature {
                beta: optional_f64(decoder, &["attn_scale"])?.unwrap_or(0.1),
                window: optional_usize(decoder, &["floor_scale"])?.unwrap_or(8192),
                offset: 1,
            })
        })
        .transpose()?;
    Ok(AttentionPattern {
        rotation_free,
        rotary_chunk: optional_usize(decoder, &["attention_chunk_size"])?
            .filter(|&chunk| chunk > 0),
        rotation_free_temperature: temperature,
        rotary_unit_qk_norm: optional_bool(decoder, "use_qk_norm")?.unwrap_or(false),
    })
}
