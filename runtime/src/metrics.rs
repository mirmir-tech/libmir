use std::{
    fmt,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::kv::CacheStats;

#[derive(Debug)]
pub struct GenerationMetricsRecorder {
    started: Instant,
    inspect: Duration,
    prompt: Duration,
    load: Duration,
    prefill: Duration,
    decode: Duration,
    sampling: Duration,
    prompt_tokens: usize,
    prefill_tokens: usize,
    generated_tokens: usize,
    decode_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub durations_ms: GenerationDurationsMs,
    pub tokens: GenerationTokenCounts,
    pub throughput: GenerationThroughput,
    pub kv_cache: CacheStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationDurationsMs {
    pub total: f64,
    pub active: f64,
    pub inspect: f64,
    pub prompt: f64,
    pub load: f64,
    pub prefill: f64,
    pub decode: f64,
    pub sampling: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationTokenCounts {
    pub prompt: usize,
    pub prefill: usize,
    pub generated: usize,
    pub decode_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationThroughput {
    pub prefill: ThroughputRate,
    pub decode: ThroughputRate,
    pub generated: ThroughputRate,
    pub generated_active: ThroughputRate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThroughputRate {
    pub per_second: Option<f64>,
    pub unit: ThroughputUnit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputUnit {
    Token,
}

impl GenerationMetricsRecorder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            inspect: Duration::ZERO,
            prompt: Duration::ZERO,
            load: Duration::ZERO,
            prefill: Duration::ZERO,
            decode: Duration::ZERO,
            sampling: Duration::ZERO,
            prompt_tokens: 0,
            prefill_tokens: 0,
            generated_tokens: 0,
            decode_steps: 0,
        }
    }

    pub fn record_inspect(&mut self, duration: Duration) {
        self.inspect += duration;
    }

    pub fn record_prompt(&mut self, duration: Duration, tokens: usize) {
        self.prompt += duration;
        self.prompt_tokens = tokens;
    }

    pub fn record_load(&mut self, duration: Duration) {
        self.load += duration;
    }

    pub fn record_prefill(&mut self, duration: Duration, tokens: usize) {
        self.prefill += duration;
        self.prefill_tokens = tokens;
    }

    pub fn record_decode(&mut self, duration: Duration) {
        self.decode += duration;
        self.decode_steps += 1;
    }

    pub fn record_sampling(&mut self, duration: Duration) {
        self.sampling += duration;
    }

    pub fn record_generated(&mut self, tokens: usize) {
        self.generated_tokens = tokens;
    }

    #[must_use]
    pub fn snapshot(&self, cache: CacheStats) -> GenerationMetrics {
        let total = self.started.elapsed();
        let active = self.active_duration();
        GenerationMetrics {
            durations_ms: GenerationDurationsMs {
                total: ms(total),
                active: ms(active),
                inspect: ms(self.inspect),
                prompt: ms(self.prompt),
                load: ms(self.load),
                prefill: ms(self.prefill),
                decode: ms(self.decode),
                sampling: ms(self.sampling),
            },
            tokens: GenerationTokenCounts {
                prompt: self.prompt_tokens,
                prefill: self.prefill_tokens,
                generated: self.generated_tokens,
                decode_steps: self.decode_steps,
            },
            throughput: GenerationThroughput {
                prefill: ThroughputRate::new(
                    self.prefill_tokens,
                    self.prefill,
                    ThroughputUnit::Token,
                ),
                decode: ThroughputRate::new(self.decode_steps, self.decode, ThroughputUnit::Token),
                generated: ThroughputRate::new(self.generated_tokens, total, ThroughputUnit::Token),
                generated_active: ThroughputRate::new(
                    self.generated_tokens,
                    active,
                    ThroughputUnit::Token,
                ),
            },
            kv_cache: cache,
        }
    }

    fn active_duration(&self) -> Duration {
        self.prefill + self.decode + self.sampling
    }
}

impl Default for GenerationMetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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

impl fmt::Display for ThroughputRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.per_second {
            Some(rate) => write!(formatter, "{rate:.3} {}", self.unit),
            None => formatter.write_str("n/a"),
        }
    }
}

impl fmt::Display for ThroughputUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Token => "tok/s",
        })
    }
}
