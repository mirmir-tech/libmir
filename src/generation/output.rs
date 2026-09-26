use models::{
    generation::{GenerationChannel, GenerationSettings, GenerationToken, OutputNormalizer},
    tokenizer::{TextDecoder, TextTokenizer},
};
use runtime::metrics::{GenerationMetrics, GenerationMetricsRecorder};

use super::telemetry::trace_latency;
use crate::{Error, Session};

pub(super) fn prepare<'a>(
    descriptor: &'a crate::ModelDescriptor,
    prepared: &super::PreparedGeneration,
    request: &super::GenerationRequest,
) -> crate::Result<(TokenStream<'a>, usize)> {
    let tokenizer = descriptor.tokenizer();
    let decoder = descriptor.decoder().ok_or_else(missing_decoder)?;
    let vocab_size = tokenizer.vocab_size().min(decoder.vocab_size);
    Ok((TokenStream::new(tokenizer, prepared.normalizer(tokenizer, request)), vocab_size))
}

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

/// Incremental detokenization joined with channel normalization.
pub(super) struct TokenStream<'a> {
    decoder: TextDecoder<'a>,
    normalizer: OutputNormalizer,
}

impl<'a> TokenStream<'a> {
    pub(super) fn new(tokenizer: &'a TextTokenizer, normalizer: OutputNormalizer) -> Self {
        Self { decoder: tokenizer.decoder(), normalizer }
    }

    pub(super) fn step(&mut self, token: u32) -> crate::Result<Option<GenerationToken>> {
        let piece = self.decoder.step(token)?.unwrap_or_default();
        Ok(self.normalizer.push(token, piece))
    }

    /// Releases text withheld as a partial UTF-8 sequence with its ids.
    pub(super) fn finish(&mut self) -> crate::Result<Option<GenerationToken>> {
        Ok(self.decoder.finish()?.and_then(|piece| self.normalizer.finish(piece)))
    }

    pub(super) fn finish_into(
        &mut self,
        text: &mut String,
        reasoning: &mut String,
        tool_calls: &mut String,
        token: &mut dyn FnMut(GenerationToken),
    ) -> crate::Result<()> {
        if let Some(delta) = self.finish()? {
            append_delta(&delta, text, reasoning, tool_calls);
            token(delta);
        }
        Ok(())
    }
}
