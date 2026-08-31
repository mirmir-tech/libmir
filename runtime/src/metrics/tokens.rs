use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationTokenCounts {
    pub prompt: usize,
    pub prefill: usize,
    pub generated: usize,
    pub decode_steps: usize,
    pub first_published_after_tokens: usize,
    #[serde(default)]
    pub recovery_attempts: usize,
    #[serde(default)]
    pub recovery_tokens: usize,
    #[serde(default)]
    pub reasoning_exits: usize,
    #[serde(default)]
    pub reasoning_exit_tokens: usize,
}
