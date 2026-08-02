use std::time::Duration;

use ::runtime::kv::BlockTable;
use mircuda::{DeviceBuffer, bf16};

use self::measure::{measure_direct, measure_split};
use super::{PagedAttentionBf16, PagedKvCache};
use crate::{
    AttentionExecution, PlanSource, Result,
    backend::tuning::{AttentionFamily, AttentionProfileRequest},
};

mod measure;
mod selection;

pub use measure::{candidate_partitions, sample_contexts};
pub use selection::{SplitMeasurement, execution_average, select_execution};

impl PagedAttentionBf16 {
    pub(super) fn ensure_tuned(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) {
        if self.tuning_complete {
            return;
        }
        let request = AttentionProfileRequest {
            family: AttentionFamily::Paged,
            plan: self.plan_request,
            block_size: self.spec.block_size,
            dtype: self.spec.dtype,
            window_tokens: window,
        };
        if !self.profile_allowed {
            self.tuning_complete = true;
            return;
        }
        if let Some((execution, source)) = self.tuner.lookup_attention(request) {
            if let Err(error) = self.apply_execution(execution) {
                tracing::warn!(?execution, %error, "discarded unavailable CUDA attention profile");
            } else {
                trace_selection(request, execution, source, None);
            }
            self.tuning_complete = true;
            return;
        }
        if !self.tuner.prepares_candidates(PlanSource::Heuristic)
            || !self.tuner.claim_attention(request)
        {
            self.tuning_complete = true;
            return;
        }
        match self.measure_attention(query, cache, table, output, window, scale) {
            Ok((execution, average, elapsed)) => {
                if let Err(error) = self.apply_execution(execution) {
                    self.tuner.abandon_attention(request);
                    tracing::warn!(?execution, %error, "failed to apply CUDA attention tuning");
                } else {
                    self.tuner.record_attention(request, execution, average, elapsed);
                    trace_selection(request, execution, PlanSource::MeasuredStartup, Some(average));
                }
            },
            Err(error) => {
                self.tuner.abandon_attention(request);
                tracing::warn!(%error, "CUDA attention tuning retained its fallback");
            },
        }
        self.tuning_complete = true;
    }

    fn measure_attention(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) -> Result<(AttentionExecution, Duration, Duration)> {
        self.validate(cache, table)?;
        self.update_table(table)?;
        let token_count = table.token_len();
        let visible = window.map_or(token_count, |limit| token_count.min(limit));
        let contexts = sample_contexts(visible, self.spec.block_size);
        let (direct, mut elapsed) =
            measure_direct(self, query, cache, table, output, window, scale, &contexts)?;
        let mut splits = Vec::new();
        for partition_tokens in candidate_partitions(self.partition_tokens) {
            match measure_split(
                self, partition_tokens, query, cache, table, output, window, scale, &contexts,
            ) {
                Ok((timings, consumed)) => {
                    let score = timings.iter().copied().sum();
                    elapsed = elapsed.saturating_add(consumed);
                    splits.push(SplitMeasurement { partition_tokens, timings, score });
                },
                Err(error) => tracing::debug!(
                    partition_tokens,
                    %error,
                    "discarded unavailable CUDA split-attention tuning candidate"
                ),
            }
        }
        let selected = select_execution(
            self.fallback_execution,
            self.partition_tokens,
            self.plan_request.max_context_tokens,
            &contexts,
            &direct,
            &splits,
            self.tuner.minimum_improvement_bps(),
        );
        let average = execution_average(selected, &direct, &splits);
        Ok((selected, average, elapsed))
    }

    fn apply_execution(&mut self, execution: AttentionExecution) -> Result<()> {
        let (partition_tokens, threshold_tokens) = match execution {
            AttentionExecution::Direct => {
                (self.partition_tokens, self.plan_request.max_context_tokens + 1)
            },
            AttentionExecution::SplitKv { partition_tokens, threshold_tokens } => {
                (partition_tokens, threshold_tokens)
            },
        };
        if partition_tokens != self.partition_tokens {
            let split = crate::kernels::SplitPagedAttention::compile(
                &self.backend.inner.compiler,
                self.spec,
                partition_tokens,
            )?;
            let (values, statistics) = split.workspace_lengths();
            let workspace = crate::kernels::SplitAttentionWorkspace::new(
                self.pool.allocate(&self.stream, values)?,
                self.pool.allocate(&self.stream, statistics)?,
                self.pool.allocate(&self.stream, statistics)?,
            );
            self.split = split;
            self.split_workspace = workspace;
            self.partition_tokens = partition_tokens;
        }
        self.split_threshold = u32::try_from(threshold_tokens)?;
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
        query_heads = request.plan.query_heads,
        kv_heads = request.plan.kv_heads,
        head_dim = request.plan.head_dim,
        value_head_dim = request.plan.value_head_dim,
        max_context_tokens = request.plan.max_context_tokens,
        block_size = request.block_size,
        window_tokens = request.window_tokens,
        ?execution,
        ?source,
        average_us = average.map(|duration| duration.as_micros()),
        "selected CUDA paged-attention execution"
    );
}
