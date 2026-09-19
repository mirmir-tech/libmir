use std::collections::VecDeque;

use uuid::Uuid;

/// Admission preference for cached continuations during resident decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachedPrefillPolicy {
    /// Preserve the backend's normal prefill/decode scheduling.
    #[default]
    BackendDefault,
    /// Admit one cached queue head with at most one block of work per step.
    /// This lowers refill first-token latency at the cost of resident decode
    /// latency.
    InterleaveOneBlock,
}

impl std::fmt::Display for CachedPrefillPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BackendDefault => "backend_default",
            Self::InterleaveOneBlock => "interleave_one_block",
        })
    }
}

/// When newly admitted prompts may begin decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillDecodePolicy {
    /// Preserve the backend's latency/throughput scheduling policy.
    #[default]
    BackendDefault,
    /// Finish a bounded admitted cohort before publishing its first tokens.
    /// Improves decode occupancy at the cost of early-request first-token
    /// latency.
    CompleteCohort,
}

impl std::fmt::Display for PrefillDecodePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BackendDefault => "backend_default",
            Self::CompleteCohort => "complete_cohort",
        })
    }
}

/// Admission of new short prompts during an existing long prefill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillRefillPolicy {
    /// Keep the admitted prefill batch closed.
    #[default]
    Closed,
    /// Join queued prompts of at most 128 tokens at CUDA chunk boundaries.
    /// Requires a mixed-attention runner with combined steps and interleaved
    /// decode; reduces short-request waiting at the cost of long-request
    /// first-token latency.
    ShortPrompt,
}

impl std::fmt::Display for PrefillRefillPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Closed => "closed",
            Self::ShortPrompt => "short_prompt",
        })
    }
}

#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub id: Uuid,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_requests: usize,
    pub max_batch_tokens: usize,
    pub prefill_batch_wait_us: u64,
    pub decode_batch_wait_us: u64,
    pub decode_priority_burst: usize,
    pub cached_prefill_policy: CachedPrefillPolicy,
    pub prefill_decode_policy: PrefillDecodePolicy,
    pub prefill_refill_policy: PrefillRefillPolicy,
}

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub requests: Vec<ScheduledRequest>,
    pub token_budget: usize,
}

#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<ScheduledRequest>,
}

impl Scheduler {
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self { config, waiting: VecDeque::new() }
    }

    pub fn push(&mut self, request: ScheduledRequest) {
        self.waiting.push_back(request);
    }

    #[must_use]
    pub fn pop_next(&mut self) -> Option<ScheduledRequest> {
        self.waiting.pop_front()
    }

    #[must_use]
    pub fn pop_batch(&mut self) -> ScheduledBatch {
        let mut requests = Vec::new();
        let mut token_budget = 0;
        while requests.len() < self.config.max_batch_requests {
            let Some(next) = self.waiting.front() else {
                break;
            };
            let request_tokens = next.prompt_tokens + next.max_tokens;
            if !requests.is_empty() && token_budget + request_tokens > self.config.max_batch_tokens
            {
                break;
            }
            let Some(next) = self.waiting.pop_front() else {
                break;
            };
            token_budget += request_tokens;
            requests.push(next);
        }
        ScheduledBatch { requests, token_budget }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_requests: 16,
            max_batch_tokens: 8192,
            prefill_batch_wait_us: 200,
            decode_batch_wait_us: 200,
            decode_priority_burst: 8,
            cached_prefill_policy: CachedPrefillPolicy::default(),
            prefill_decode_policy: PrefillDecodePolicy::default(),
            prefill_refill_policy: PrefillRefillPolicy::default(),
        }
    }
}
