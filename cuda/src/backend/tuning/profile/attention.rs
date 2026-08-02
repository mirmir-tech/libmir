use std::{collections::HashMap, time::Duration};

use super::{
    AttentionProfileRequest, AttentionRuntimeEntry, CudaAutoTuner, storage::StoredAttentionEntry,
};
use crate::{AttentionExecution, PlanSource};

impl CudaAutoTuner {
    pub(crate) fn lookup_attention(
        &self,
        request: AttentionProfileRequest,
    ) -> Option<(AttentionExecution, PlanSource)> {
        if self.inner.config.mode == super::CudaTuningMode::Disabled {
            return None;
        }
        self.inner
            .state
            .lock()
            .ok()?
            .attention
            .get(&request)
            .map(|entry| (entry.execution, entry.source))
    }

    pub(crate) fn claim_attention(&self, request: AttentionProfileRequest) -> bool {
        let Ok(mut state) = self.inner.state.lock() else {
            return false;
        };
        self.inner.config.mode == super::CudaTuningMode::Startup
            && !state.sealed
            && state.budget.available()
            && !state.attention.contains_key(&request)
            && state.attention_inflight.insert(request)
    }

    pub(crate) fn record_attention(
        &self,
        request: AttentionProfileRequest,
        execution: AttentionExecution,
        average: Duration,
        tuning_elapsed: Duration,
    ) {
        let snapshot = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            state.attention_inflight.remove(&request);
            state.budget.consume(tuning_elapsed);
            state.attention.insert(
                request,
                AttentionRuntimeEntry {
                    execution,
                    source: PlanSource::MeasuredStartup,
                    average_ns: u64::try_from(average.as_nanos()).unwrap_or(u64::MAX),
                },
            );
            Self::snapshot(&state)
        };
        self.persist(snapshot);
    }

    pub(crate) fn abandon_attention(&self, request: AttentionProfileRequest) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.attention_inflight.remove(&request);
        }
    }
}

pub(super) fn stored_entries(
    entries: &HashMap<AttentionProfileRequest, AttentionRuntimeEntry>,
) -> Vec<StoredAttentionEntry> {
    let mut stored = entries
        .iter()
        .map(|(request, entry)| StoredAttentionEntry {
            request: *request,
            execution: entry.execution,
            average_ns: entry.average_ns,
        })
        .collect::<Vec<_>>();
    stored.sort_by_key(|entry| {
        (
            entry.request.family as u8,
            entry.request.plan.query_heads,
            entry.request.plan.kv_heads,
            entry.request.plan.head_dim,
            entry.request.plan.value_head_dim,
            entry.request.plan.max_context_tokens,
            entry.request.block_size,
            entry.request.window_tokens,
        )
    });
    stored
}
