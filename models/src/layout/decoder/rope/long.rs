//! `LongRoPE` (Phi-3 `rope_type = longrope`): each rotary pair's period is
//! multiplied by a declared factor, and the cosine and sine by an attention
//! factor. Like MLX-LM `SuScaledRoPE`, the long factors apply at every
//! position, as llama.cpp applies them to any context beyond the original.

use std::sync::Mutex;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::{RopeScaling, integer, optional_number};
use crate::{error::Result, layout::decoder::parse::invalid};

/// Per-pair period factors. Each distinct list is interned once for the
/// process, so scaling stays `Copy` and reloading a model reuses its list.
#[derive(Clone, Copy, PartialEq)]
pub struct RopeFactors(&'static [f32]);

static INTERNED: Mutex<Vec<&'static [f32]>> = Mutex::new(Vec::new());

impl RopeFactors {
    pub fn new(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid("LongRoPE needs at least one factor"));
        }
        let values = values
            .iter()
            .map(|&value| {
                // The factors are `f32` in GGUF and in the checkpoints' JSON.
                #[allow(clippy::cast_possible_truncation)]
                let value = value as f32;
                if value.is_finite() && value > 0.0 {
                    Ok(value)
                } else {
                    Err(invalid("LongRoPE factors must be positive and finite"))
                }
            })
            .collect::<Result<Vec<f32>>>()?;
        Ok(Self(intern(values)))
    }

    #[must_use]
    pub const fn as_slice(&self) -> &[f32] {
        self.0
    }
}

fn intern(values: Vec<f32>) -> &'static [f32] {
    let mut interned = INTERNED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&known) = interned.iter().find(|known| known[..] == values[..]) {
        return known;
    }
    let leaked: &'static [f32] = Box::leak(values.into_boxed_slice());
    interned.push(leaked);
    drop(interned);
    leaked
}

impl std::fmt::Debug for RopeFactors {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.as_slice()).finish()
    }
}

impl Serialize for RopeFactors {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RopeFactors {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let values = Vec::<f64>::deserialize(deserializer)?;
        Self::new(&values).map_err(serde::de::Error::custom)
    }
}

/// `rope_scaling` of type `longrope`; Phi-3 declares the original context at
/// the top level, and the attention factor (`attention_factor` or
/// `long_mscale`) defaults to
/// `sqrt(1 + ln(max / original) / ln(original))` (transformers
/// `_compute_longrope_parameters`).
pub(super) fn long_rope(config: &Value, value: &Value) -> Result<RopeScaling> {
    let factors = value
        .get("long_factor")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("rope_scaling.long_factor must be an array"))?
        .iter()
        .map(|factor| factor.as_f64().ok_or_else(|| invalid("LongRoPE factors must be numbers")))
        .collect::<Result<Vec<_>>>()?;
    let original_context_len = integer(value, "original_max_position_embeddings")
        .or_else(|_| integer(config, "original_max_position_embeddings"))?;
    let maximum = integer(config, "max_position_embeddings")?;
    // Phi-3.5-MoE declares the long factors' scale as `long_mscale`, which
    // MLX-LM `SuScaledRoPE` applies.
    let declared = match optional_number(value, "attention_factor")? {
        Some(factor) => Some(factor),
        None => optional_number(value, "long_mscale")?,
    };
    let attention_factor = match declared {
        Some(factor) => factor,
        None if maximum > original_context_len => {
            let ratio = f64::from(u32::try_from(maximum)?)
                / f64::from(u32::try_from(original_context_len)?);
            (1.0 + ratio.log(f64::from(u32::try_from(original_context_len)?))).sqrt()
        },
        None => 1.0,
    };
    if original_context_len == 0 || !attention_factor.is_finite() || attention_factor <= 0.0 {
        return Err(invalid("invalid LongRoPE rope_scaling parameters"));
    }
    Ok(RopeScaling::LongRope {
        factors: RopeFactors::new(&factors)?,
        original_context_len,
        attention_factor,
    })
}
