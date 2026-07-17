use std::collections::VecDeque;

use uuid::Uuid;

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
    pub decode_batch_wait_us: u64,
    pub decode_priority_burst: usize,
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
            decode_batch_wait_us: 200,
            decode_priority_burst: 8,
        }
    }
}
