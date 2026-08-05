use std::time::Duration;

use mircuda::{DeviceBuffer, bf16};

use self::measure::measure;
use super::{BatchedPagedAttentionBf16, allocate_workspace, threshold};
use crate::{
    AttentionExecution, PlanSource, Result,
    backend::{
        PagedDecodeBatch, PagedKvCache,
        tuning::{AttentionFamily, AttentionProfileRequest},
    },
    kernels::{BatchedSplitPagedAttention, PagedAttentionSpec},
};

mod measure;

impl BatchedPagedAttentionBf16 {
    pub(super) fn ensure_tuned(
        &mut self,
        query: &DeviceBuffer<bf16>,
        cache: &PagedKvCache,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
        window: Option<usize>,
        scale: f32,
    ) {
        if self.tuning_complete {
            return;
        }
        let request = self.profile_request(window);
        if !self.profile_allowed {
            self.tuning_complete = true;
            return;
        }
        if let Some((execution, source)) = self.backend.inner.tuner.lookup_attention(request) {
            self.apply_or_warn(request, execution, source, None);
            return;
        }
        if !self.backend.inner.tuner.claim_dynamic_attention(request) {
            self.tuning_complete = true;
            return;
        }
        match measure(self, query, cache, batch, output, window, scale) {
            Ok((execution, average, elapsed)) => {
                if self.apply_execution(execution).is_ok() {
                    self.backend.inner.tuner.record_attention(request, execution, average, elapsed);
                    trace_selection(request, execution, PlanSource::MeasuredStartup, Some(average));
                } else {
                    self.backend.inner.tuner.abandon_attention(request);
                }
            },
            Err(error) => {
                self.backend.inner.tuner.abandon_attention(request);
                tracing::warn!(%error, "CUDA batched attention tuning retained its fallback");
            },
        }
        self.tuning_complete = true;
    }

    fn apply_or_warn(
        &mut self,
        request: AttentionProfileRequest,
        execution: AttentionExecution,
        source: PlanSource,
        average: Option<Duration>,
    ) {
        if let Err(error) = self.apply_execution(execution) {
            tracing::warn!(?execution, %error, "discarded unavailable CUDA batch attention profile");
        } else {
            trace_selection(request, execution, source, average);
        }
        self.tuning_complete = true;
    }

    fn profile_request(&self, window: Option<usize>) -> AttentionProfileRequest {
        AttentionProfileRequest {
            family: AttentionFamily::Paged,
            plan: self.plan_request,
            batch_rows: self.max_batch,
            block_size: self.storage.cache.block_size,
            dtype: self.storage.cache.dtype,
            window_tokens: window,
        }
    }

    fn apply_execution(&mut self, execution: AttentionExecution) -> Result<()> {
        let partition = match execution {
            AttentionExecution::Direct => self.split.partition_tokens(),
            AttentionExecution::SplitKv { partition_tokens, .. } => partition_tokens,
        };
        if partition != self.split.partition_tokens() {
            let split = BatchedSplitPagedAttention::compile(
                &self.backend.inner.compiler,
                self.spec(),
                self.max_batch,
                partition,
            )?;
            let (values, statistics) = split.workspace_lengths();
            if self.split_workspace.values.len() < values
                || self.split_workspace.maxima.len() < statistics
                || self.split_workspace.denominators.len() < statistics
            {
                self.split_workspace = allocate_workspace(&self.backend, &split)?;
            }
            self.split = split;
        }
        self.split_threshold = threshold(execution, self.plan_request.max_context_tokens);
        Ok(())
    }

    pub(super) fn spec(&self) -> PagedAttentionSpec {
        PagedAttentionSpec {
            block_size: self.storage.cache.block_size,
            max_blocks: self.max_blocks,
            query_heads: self.plan_request.query_heads,
            kv_heads: self.storage.kv_heads,
            head_dim: self.storage.key_head_dim,
            value_head_dim: self.storage.value_head_dim,
            dtype: self.storage.cache.dtype,
        }
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
        batch_rows = request.batch_rows,
        dtype = %request.dtype,
        ?execution,
        ?source,
        average_us = average.map(|duration| duration.as_micros()),
        "selected CUDA batched paged-attention execution"
    );
}
