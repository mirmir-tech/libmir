mod display;
mod recorder;
mod throughput;
mod tokens;

use std::time::{Duration, Instant};

use recorder::ms;
use serde::{Deserialize, Serialize};
pub use tokens::GenerationTokenCounts;

use crate::kv::CacheStats;

#[derive(Debug)]
pub struct GenerationMetricsRecorder {
    started: Instant,
    inspect: Duration,
    prompt: Duration,
    prompt_render: Duration,
    tokenize: Duration,
    output_setup: Duration,
    sampler_setup: Duration,
    session_setup: Duration,
    load: Duration,
    prefill: Duration,
    cache_prepare: Duration,
    scheduler_wait: Duration,
    backend_wait: Duration,
    backend_prefill: Duration,
    first_token_publish: Duration,
    first_token_total: Duration,
    decode: Duration,
    sampling: Duration,
    prompt_tokens: usize,
    prefill_tokens: usize,
    generated_tokens: usize,
    decode_steps: usize,
    first_published_after_tokens: usize,
    recovery_attempts: usize,
    recovery_tokens: usize,
    reasoning_exits: usize,
    reasoning_exit_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub durations_ms: GenerationDurationsMs,
    pub tokens: GenerationTokenCounts,
    pub throughput: GenerationThroughput,
    pub kv_cache: CacheStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationDurationsMs {
    pub total: f64,
    pub active: f64,
    pub inspect: f64,
    pub prompt: f64,
    pub prompt_render: f64,
    pub tokenize: f64,
    pub output_setup: f64,
    pub sampler_setup: f64,
    pub session_setup: f64,
    pub load: f64,
    pub prefill: f64,
    pub cache_prepare: f64,
    pub scheduler_wait: f64,
    pub backend_wait: f64,
    pub backend_prefill: f64,
    pub first_token_publish: f64,
    pub first_token_total: f64,
    pub decode: f64,
    pub sampling: f64,
}

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

impl GenerationMetricsRecorder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            inspect: Duration::ZERO,
            prompt: Duration::ZERO,
            prompt_render: Duration::ZERO,
            tokenize: Duration::ZERO,
            output_setup: Duration::ZERO,
            sampler_setup: Duration::ZERO,
            session_setup: Duration::ZERO,
            load: Duration::ZERO,
            prefill: Duration::ZERO,
            cache_prepare: Duration::ZERO,
            scheduler_wait: Duration::ZERO,
            backend_wait: Duration::ZERO,
            backend_prefill: Duration::ZERO,
            first_token_publish: Duration::ZERO,
            first_token_total: Duration::ZERO,
            decode: Duration::ZERO,
            sampling: Duration::ZERO,
            prompt_tokens: 0,
            prefill_tokens: 0,
            generated_tokens: 0,
            decode_steps: 0,
            first_published_after_tokens: 0,
            recovery_attempts: 0,
            recovery_tokens: 0,
            reasoning_exits: 0,
            reasoning_exit_tokens: 0,
        }
    }

    pub fn record_inspect(&mut self, duration: Duration) {
        self.inspect += duration;
    }

    pub fn record_prompt(&mut self, duration: Duration, tokens: usize) {
        self.prompt += duration;
        self.prompt_tokens = tokens;
    }

    pub fn record_prompt_stages(&mut self, render: Duration, tokenize: Duration) {
        self.prompt_render += render;
        self.tokenize += tokenize;
    }

    pub fn record_setup_stages(&mut self, output: Duration, sampler: Duration, session: Duration) {
        self.output_setup += output;
        self.sampler_setup += sampler;
        self.session_setup += session;
    }

    pub fn record_load(&mut self, duration: Duration) {
        self.load += duration;
    }

    pub fn record_prefill(&mut self, duration: Duration, tokens: usize) {
        self.prefill += duration;
        self.prefill_tokens = tokens;
    }

    pub fn record_prefill_stages(
        &mut self,
        cache_prepare: Duration,
        scheduler_wait: Duration,
        backend_wait: Duration,
        backend_prefill: Duration,
    ) {
        self.cache_prepare += cache_prepare;
        self.scheduler_wait += scheduler_wait;
        self.backend_wait += backend_wait;
        self.backend_prefill += backend_prefill;
    }

    pub fn record_first_token_publish(&mut self, duration: Duration, generated_tokens: usize) {
        self.first_token_publish = duration;
        self.first_token_total = self.started.elapsed();
        self.first_published_after_tokens = generated_tokens;
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
                prompt_render: ms(self.prompt_render),
                tokenize: ms(self.tokenize),
                output_setup: ms(self.output_setup),
                sampler_setup: ms(self.sampler_setup),
                session_setup: ms(self.session_setup),
                load: ms(self.load),
                prefill: ms(self.prefill),
                cache_prepare: ms(self.cache_prepare),
                scheduler_wait: ms(self.scheduler_wait),
                backend_wait: ms(self.backend_wait),
                backend_prefill: ms(self.backend_prefill),
                first_token_publish: ms(self.first_token_publish),
                first_token_total: ms(self.first_token_total),
                decode: ms(self.decode),
                sampling: ms(self.sampling),
            },
            tokens: GenerationTokenCounts {
                prompt: self.prompt_tokens,
                prefill: self.prefill_tokens,
                generated: self.generated_tokens,
                decode_steps: self.decode_steps,
                first_published_after_tokens: self.first_published_after_tokens,
                recovery_attempts: self.recovery_attempts,
                recovery_tokens: self.recovery_tokens,
                reasoning_exits: self.reasoning_exits,
                reasoning_exit_tokens: self.reasoning_exit_tokens,
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
