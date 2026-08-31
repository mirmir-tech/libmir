use std::time::Instant;

use runtime::{
    backend::PrefillOutput,
    metrics::{GenerationMetrics, GenerationMetricsRecorder},
};

pub(super) fn record_publish(
    metrics: &mut GenerationMetricsRecorder,
    started: Instant,
    generated_tokens: usize,
    published: &mut bool,
) {
    if !*published {
        metrics.record_first_token_publish(started.elapsed(), generated_tokens);
        *published = true;
    }
}

pub(super) fn record_prefill_metrics(
    metrics: &mut GenerationMetricsRecorder,
    started: Instant,
    prompt_tokens: usize,
    output: &PrefillOutput,
) {
    metrics.record_prefill(started.elapsed(), prompt_tokens);
    if let Some(timings) = output.timings {
        metrics.record_prefill_stages(
            timings.cache_prepare,
            timings.scheduler_queue,
            timings.backend_wait,
            timings.backend_execution,
        );
    }
}

pub(super) fn trace_latency(metrics: &GenerationMetrics) {
    let durations = &metrics.durations_ms;
    tracing::info!(
        prompt_render_ms = durations.prompt_render,
        tokenize_ms = durations.tokenize,
        prompt_prepare_ms = durations.prompt,
        output_setup_ms = durations.output_setup,
        sampler_setup_ms = durations.sampler_setup,
        session_setup_ms = durations.session_setup,
        cache_prepare_ms = durations.cache_prepare,
        scheduler_wait_ms = durations.scheduler_wait,
        backend_wait_ms = durations.backend_wait,
        backend_prefill_ms = durations.backend_prefill,
        prefill_total_ms = durations.prefill,
        first_token_publish_ms = durations.first_token_publish,
        first_token_total_ms = durations.first_token_total,
        first_published_after_tokens = metrics.tokens.first_published_after_tokens,
        decode_ms = durations.decode,
        decode_steps = metrics.tokens.decode_steps,
        decode_tokens_per_second = metrics.throughput.decode.per_second,
        sampling_ms = durations.sampling,
        generation_total_ms = durations.total,
        "generation latency breakdown"
    );
}
