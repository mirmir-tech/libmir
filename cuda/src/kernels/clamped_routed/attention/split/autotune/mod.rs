use std::time::Duration;

use mircuda::{DeviceBuffer, Stream, bf16};

use self::measure::{measure_direct, measure_split};
use super::ClampedRoutedSplitDecode;
use crate::{
    AttentionExecution, PlanSource, Result,
    backend::{
        AttentionProfileRequest, AttentionSplitMeasurement, attention_execution_average,
        candidate_partitions, sample_contexts, select_attention_execution,
    },
    kernels::{ClampedRoutedAttention, SplitAttentionWorkspace, SplitPagedAttention},
};

mod measure;

impl ClampedRoutedSplitDecode {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ensure_tuned(
        &mut self,
        direct: &ClampedRoutedAttention,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        table: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
        blocks: usize,
        window: Option<usize>,
        scale: f32,
    ) {
        if self.tuning_complete {
            return;
        }
        let mut request = self.request;
        request.window_tokens = window;
        let tuner = self.backend.auto_tuner().clone();
        if !self.profile_allowed {
            self.tuning_complete = true;
            return;
        }
        if let Some((execution, source)) = tuner.lookup_attention(request) {
            if let Err(error) = self.apply_execution(execution) {
                tracing::warn!(?execution, %error, "discarded unavailable sink-attention profile");
            } else {
                trace_selection(request, execution, source, None);
            }
            self.tuning_complete = true;
            return;
        }
        if !tuner.prepares_candidates(PlanSource::Heuristic) || !tuner.claim_attention(request) {
            self.tuning_complete = true;
            return;
        }
        match self.measure_attention(
            direct, stream, query, key_pages, value_pages, table, sinks, output, tokens, blocks,
            window, scale,
        ) {
            Ok((execution, average, elapsed)) => {
                if let Err(error) = self.apply_execution(execution) {
                    tuner.abandon_attention(request);
                    tracing::warn!(?execution, %error, "failed to apply sink-attention tuning");
                } else {
                    tuner.record_attention(request, execution, average, elapsed);
                    trace_selection(request, execution, PlanSource::MeasuredStartup, Some(average));
                }
            },
            Err(error) => {
                tuner.abandon_attention(request);
                tracing::warn!(%error, "sink-attention tuning retained its fallback");
            },
        }
        self.tuning_complete = true;
    }

    #[allow(clippy::too_many_arguments)]
    fn measure_attention(
        &self,
        direct: &ClampedRoutedAttention,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        table: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
        blocks: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<(AttentionExecution, Duration, Duration)> {
        let visible = window.map_or(tokens, |limit| tokens.min(limit));
        let contexts = sample_contexts(visible, self.request.block_size);
        let (direct_timings, mut elapsed) = measure_direct(
            self, direct, stream, query, key_pages, value_pages, table, sinks, output, blocks,
            window, scale, &contexts,
        )?;
        let mut splits = Vec::new();
        for partition_tokens in candidate_partitions(self.partition_tokens) {
            match measure_split(
                self, partition_tokens, stream, query, key_pages, value_pages, table, sinks,
                output, blocks, window, scale, &contexts,
            ) {
                Ok((timings, consumed)) => {
                    let score = timings.iter().copied().sum();
                    elapsed = elapsed.saturating_add(consumed);
                    splits.push(AttentionSplitMeasurement { partition_tokens, timings, score });
                },
                Err(error) => tracing::debug!(
                    partition_tokens,
                    %error,
                    "discarded unavailable sink-attention candidate"
                ),
            }
        }
        let selected = select_attention_execution(
            self.fallback,
            self.partition_tokens,
            self.request.plan.max_context_tokens,
            &contexts,
            &direct_timings,
            &splits,
            self.backend.auto_tuner().minimum_improvement_bps(),
        );
        Ok((
            selected,
            attention_execution_average(selected, &direct_timings, &splits),
            elapsed,
        ))
    }

    fn apply_execution(&mut self, execution: AttentionExecution) -> Result<()> {
        let (partition_tokens, threshold_tokens) = match execution {
            AttentionExecution::Direct => {
                (self.partition_tokens, self.request.plan.max_context_tokens + 1)
            },
            AttentionExecution::SplitKv { partition_tokens, threshold_tokens } => {
                (partition_tokens, threshold_tokens)
            },
        };
        if partition_tokens != self.partition_tokens {
            let operation = SplitPagedAttention::compile(
                self.backend.compiler(),
                self.operation.spec(),
                partition_tokens,
            )?;
            let (values, statistics) = operation.workspace_lengths();
            self.workspace = SplitAttentionWorkspace::new(
                self.backend.pool().allocate(self.backend.stream(), values)?,
                self.backend.pool().allocate(self.backend.stream(), statistics)?,
                self.backend.pool().allocate(self.backend.stream(), statistics)?,
            );
            self.operation = operation;
            self.partition_tokens = partition_tokens;
        }
        self.threshold = threshold_tokens;
        Ok(())
    }
}

fn trace_selection(
    request: AttentionProfileRequest,
    execution: AttentionExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        family = ?request.family,
        query_heads = request.plan.query_heads,
        kv_heads = request.plan.kv_heads,
        head_dim = request.plan.head_dim,
        max_context_tokens = request.plan.max_context_tokens,
        window_tokens = request.window_tokens,
        ?execution,
        ?source,
        average_us = average.map(|duration| duration.as_micros()),
        "selected CUDA sink-attention execution"
    );
}
