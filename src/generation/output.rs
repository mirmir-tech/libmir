use models::generation::{GenerationChannel, GenerationSettings, GenerationToken};
use runtime::metrics::{GenerationMetrics, GenerationMetricsRecorder};

use super::telemetry::trace_latency;
use crate::{Error, Session};

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

pub(super) fn finish_metrics(
    metrics: &mut GenerationMetricsRecorder,
    generated_tokens: usize,
    session: &Session,
) -> GenerationMetrics {
    metrics.record_generated(generated_tokens);
    let metrics = metrics.snapshot(session.cache_stats());
    trace_latency(&metrics);
    metrics
}

pub(super) fn finalize_output(
    text: String,
    reasoning: String,
    tool_calls: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    finish_reason: &'static str,
    metrics: GenerationMetrics,
) -> GenerationOutput {
    let finish_reason = if tool_calls.is_empty() {
        finish_reason
    } else {
        "tool_calls"
    };
    GenerationOutput {
        text,
        reasoning,
        tool_calls,
        token_ids,
        prompt_tokens,
        finish_reason,
        metrics,
    }
}

pub(super) fn append_delta(
    delta: &GenerationToken,
    text: &mut String,
    reasoning: &mut String,
    tool_calls: &mut String,
) {
    match delta.channel {
        GenerationChannel::Content => text.push_str(&delta.text),
        GenerationChannel::Reasoning => reasoning.push_str(&delta.text),
        GenerationChannel::ToolCalls => tool_calls.push_str(&delta.text),
    }
}

pub(super) fn missing_decoder() -> Error {
    Error::TaskMismatch {
        requested: "generation",
        actual: "sequence scoring",
    }
}

pub(super) fn should_stop(
    settings: GenerationSettings,
    generated_tokens: usize,
    token: u32,
    stop_tokens: &[u32],
) -> bool {
    !settings.ignore_eos && generated_tokens >= settings.min_tokens && stop_tokens.contains(&token)
}
