mod output;

use std::fs;

pub use output::{GenerationChannel, GenerationToken, OutputNormalizer};
use serde_json::Value;

use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

const DEFAULT_MAX_TOKENS: usize = 2_048;
const DEFAULT_REPETITION_PENALTY: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
struct SamplingDefaults {
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

const DEFAULT_SAMPLING: SamplingDefaults =
    SamplingDefaults { temperature: 0.0, top_p: 1.0, top_k: 0 };
const DEFAULT_SAMPLE_TEMPERATURE: f32 = 1.0;

#[derive(Debug, Clone, Default)]
pub struct GenerationConfig {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    do_sample: Option<bool>,
    stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationOverrides {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationSettings {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl GenerationConfig {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        layout.generation_config_path.as_deref().map_or_else(
            || Ok(Self::default()),
            |path| Self::from_value(&serde_json::from_str(&fs::read_to_string(path)?)?),
        )
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        Ok(Self {
            max_tokens: optional_usize(value, "max_new_tokens")?,
            temperature: optional_f32(value, "temperature")?,
            top_p: optional_f32(value, "top_p")?,
            top_k: optional_usize(value, "top_k")?,
            repetition_penalty: optional_f32(value, "repetition_penalty")?,
            do_sample: optional_bool(value, "do_sample")?,
            stop_token_ids: stop_token_ids(value)?,
        })
    }

    pub fn resolve(&self, overrides: GenerationOverrides) -> Result<GenerationSettings> {
        self.resolve_with_defaults(overrides, DEFAULT_SAMPLING)
    }

    fn resolve_with_defaults(
        &self,
        overrides: GenerationOverrides,
        defaults: SamplingDefaults,
    ) -> Result<GenerationSettings> {
        let checkpoint_temperature =
            (self.do_sample != Some(false)).then_some(self.temperature).flatten();
        let temperature = overrides.temperature.or(checkpoint_temperature).unwrap_or({
            match self.do_sample {
                Some(false) => 0.0,
                Some(true) => DEFAULT_SAMPLE_TEMPERATURE,
                None => defaults.temperature,
            }
        });
        let settings = GenerationSettings {
            max_tokens: overrides.max_tokens.or(self.max_tokens).unwrap_or(DEFAULT_MAX_TOKENS),
            temperature,
            top_p: overrides.top_p.or(self.top_p).unwrap_or(defaults.top_p),
            top_k: overrides.top_k.or(self.top_k).unwrap_or(defaults.top_k),
            repetition_penalty: overrides
                .repetition_penalty
                .or(self.repetition_penalty)
                .unwrap_or(DEFAULT_REPETITION_PENALTY),
        };
        settings.validate()?;
        Ok(settings)
    }

    #[must_use]
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }
}

impl GenerationSettings {
    /// Applies request-scoped values on top of settings already resolved for
    /// the loaded model.
    pub fn with_overrides(mut self, overrides: GenerationOverrides) -> Result<Self> {
        if let Some(value) = overrides.max_tokens {
            self.max_tokens = value;
        }
        if let Some(value) = overrides.temperature {
            self.temperature = value;
        }
        if let Some(value) = overrides.top_p {
            self.top_p = value;
        }
        if let Some(value) = overrides.top_k {
            self.top_k = value;
        }
        if let Some(value) = overrides.repetition_penalty {
            self.repetition_penalty = value;
        }
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 {
            return Err(invalid("max_new_tokens must be greater than zero"));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(invalid("temperature must be finite and non-negative"));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(invalid("top_p must be finite and between zero and one"));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(invalid("repetition_penalty must be finite and positive"));
        }
        Ok(())
    }
}

fn optional_usize(value: &Value, key: &str) -> Result<Option<usize>> {
    let Some(value) = value.get(key) else {
        return Ok(None);
    };
    let number = value
        .as_u64()
        .ok_or_else(|| invalid(format!("{key} must be an unsigned integer")))?;
    Ok(Some(usize::try_from(number)?))
}

fn optional_f32(value: &Value, key: &str) -> Result<Option<f32>> {
    let Some(value) = value.get(key) else {
        return Ok(None);
    };
    let number = value.as_f64().ok_or_else(|| invalid(format!("{key} must be a number")))?;
    Ok(Some(number.to_string().parse()?))
}

fn optional_bool(value: &Value, key: &str) -> Result<Option<bool>> {
    let Some(value) = value.get(key) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| invalid(format!("{key} must be boolean")))
}

fn stop_token_ids(value: &Value) -> Result<Vec<u32>> {
    let Some(value) = value.get("eos_token_id") else {
        return Ok(Vec::new());
    };
    let values = value.as_array().map_or_else(|| std::slice::from_ref(value), Vec::as_slice);
    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        let number = value.as_u64().ok_or_else(|| invalid("eos_token_id must contain integers"))?;
        ids.push(u32::try_from(number)?);
    }
    Ok(ids)
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests;
