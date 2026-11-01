use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationThroughput {
    pub prefill: ThroughputRate,
    pub decode: ThroughputRate,
    pub generated: ThroughputRate,
    pub generated_active: ThroughputRate,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ThroughputRate {
    pub per_second: Option<f64>,
    pub unit: ThroughputUnit,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputUnit {
    #[default]
    Token,
}

impl ThroughputRate {
    #[must_use]
    pub fn new(units: usize, duration: Duration, unit: ThroughputUnit) -> Self {
        let per_second = if units == 0 || duration.is_zero() {
            None
        } else {
            units
                .to_string()
                .parse::<f64>()
                .ok()
                .map(|units| units / duration.as_secs_f64())
        };
        Self { per_second, unit }
    }
}
