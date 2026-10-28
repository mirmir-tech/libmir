use std::{collections::BTreeMap, fs, path::Path};

use serde::Deserialize;

use super::QuestionKind;
use crate::error::{ModelsError, Result};

/// Temperatures outside this range are clamped: a fitted value below one
/// sharpens instead of calibrating, and the reference runtime refuses to
/// apply one sharper than `0.5`.
const TEMPERATURE_RANGE: (f64, f64) = (0.5, 5.0);

/// `rl_agent_config.json`: sequence budgets, head depth, and calibration of a
/// Laya decision checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// Maximum tokens of one question row, state included.
    pub max_len: usize,
    /// Token budget of the instruction and its options.
    pub head_max_len: usize,
    /// Transformer layers of the decision head.
    pub head_layers: usize,
    temperatures: [f64; 3],
    by_options: BTreeMap<(QuestionKind, OptionBucket), f64>,
}

#[derive(Debug, Deserialize)]
struct RawAgentConfig {
    max_len: usize,
    head_max_len: usize,
    head_layers: usize,
    temperature: [f64; 3],
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f64>,
}

/// Option-count buckets of per-bucket temperatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OptionBucket {
    Two,
    ThreeToFive,
    SixToTen,
    ElevenOrMore,
}

impl OptionBucket {
    const fn of(options: usize) -> Self {
        match options {
            0..=2 => Self::Two,
            3..=5 => Self::ThreeToFive,
            6..=10 => Self::SixToTen,
            _ => Self::ElevenOrMore,
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "2" => Some(Self::Two),
            "3-5" => Some(Self::ThreeToFive),
            "6-10" => Some(Self::SixToTen),
            "11+" => Some(Self::ElevenOrMore),
            _ => None,
        }
    }
}

impl AgentConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawAgentConfig = serde_json::from_str(json)?;
        if raw.max_len <= raw.head_max_len {
            return Err(invalid("Laya max_len must exceed head_max_len"));
        }
        let by_options = raw
            .temperature_by_options
            .iter()
            .map(|(key, &value)| Ok((parse_bucket_key(key)?, clamp(value)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            max_len: raw.max_len,
            head_max_len: raw.head_max_len,
            head_layers: raw.head_layers,
            temperatures: [
                clamp(raw.temperature[0])?,
                clamp(raw.temperature[1])?,
                clamp(raw.temperature[2])?,
            ],
            by_options,
        })
    }

    /// Temperature dividing the logits of a question with `options` options.
    #[must_use]
    pub fn temperature(&self, kind: QuestionKind, options: usize) -> f64 {
        self.by_options
            .get(&(kind, OptionBucket::of(options)))
            .copied()
            .unwrap_or_else(|| self.temperatures[kind.index()])
    }
}

fn parse_bucket_key(key: &str) -> Result<(QuestionKind, OptionBucket)> {
    let parsed = key.split_once(':').and_then(|(kind, bucket)| {
        Some((QuestionKind::parse(kind)?, OptionBucket::parse(bucket)?))
    });
    parsed.ok_or_else(|| invalid(format!("unknown Laya temperature bucket `{key}`")))
}

fn clamp(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(invalid("Laya temperatures must be finite"));
    }
    Ok(value.clamp(TEMPERATURE_RANGE.0, TEMPERATURE_RANGE.1))
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests {
    use super::AgentConfig;
    use crate::{Result, decision::QuestionKind};

    #[test]
    fn selects_bucket_temperatures_and_clamps_sharpeners() -> Result<()> {
        let config = AgentConfig::from_json(
            r#"{"max_len": 1024, "head_max_len": 256, "head_layers": 2,
                "temperature": [1.1, 1.2, 1.3],
                "temperature_by_options": {"choice:3-5": 1.76, "choice:11+": 0.1}}"#,
        )?;
        assert!((config.temperature(QuestionKind::Choice, 4) - 1.76).abs() < 1e-12);
        assert!((config.temperature(QuestionKind::Choice, 12) - 0.5).abs() < 1e-12);
        assert!((config.temperature(QuestionKind::Choice, 2) - 1.1).abs() < 1e-12);
        assert!((config.temperature(QuestionKind::YesNo, 2) - 1.3).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn rejects_unknown_buckets() {
        let config = AgentConfig::from_json(
            r#"{"max_len": 1024, "head_max_len": 256, "head_layers": 2,
                "temperature": [1, 1, 1], "temperature_by_options": {"choice:7": 1.0}}"#,
        );
        assert!(config.is_err());
    }
}
