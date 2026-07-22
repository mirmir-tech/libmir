use runtime::metrics::GenerationMetrics;

/// Completed generation with separated output channels, tokens, and timing
/// metrics.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// User-visible assistant text.
    pub text: String,
    /// Reasoning text emitted on the reasoning channel, when present.
    pub reasoning: String,
    /// JSON emitted on the model's native tool-call channel, when present.
    pub tool_calls: String,
    /// Generated token identifiers, including a terminal stop token when
    /// emitted.
    pub token_ids: Vec<u32>,
    /// Number of tokens in the prepared prompt.
    pub prompt_tokens: usize,
    /// Stable completion reason such as `"stop"`, `"tool_calls"`, or
    /// `"max_tokens"`.
    pub finish_reason: &'static str,
    /// Prefill, decode, throughput, and cache metrics for this generation.
    pub metrics: GenerationMetrics,
}
