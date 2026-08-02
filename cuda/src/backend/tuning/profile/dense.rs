use std::{collections::HashMap, time::Duration};

use super::{CudaAutoTuner, DenseRuntimeEntry, storage::StoredDenseEntry};
use crate::{DenseExecution, DensePlanRequest, PlanSource};

impl CudaAutoTuner {
    pub(in crate::backend) fn lookup_dense(
        &self,
        request: DensePlanRequest,
    ) -> Option<(DenseExecution, PlanSource)> {
        if self.inner.config.mode == super::CudaTuningMode::Disabled {
            return None;
        }
        self.inner
            .state
            .lock()
            .ok()?
            .dense
            .get(&request)
            .map(|entry| (entry.execution, entry.source))
    }

    pub(in crate::backend) fn claim_dense(&self, request: DensePlanRequest) -> bool {
        let Ok(mut state) = self.inner.state.lock() else {
            return false;
        };
        self.inner.config.mode == super::CudaTuningMode::Startup
            && !state.sealed
            && state.budget.available()
            && !state.dense.contains_key(&request)
            && state.dense_inflight.insert(request)
    }

    pub(in crate::backend) fn record_dense(
        &self,
        request: DensePlanRequest,
        execution: DenseExecution,
        average: Duration,
        tuning_elapsed: Duration,
    ) {
        let snapshot = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            state.dense_inflight.remove(&request);
            state.budget.consume(tuning_elapsed);
            state.dense.insert(
                request,
                DenseRuntimeEntry {
                    execution,
                    source: PlanSource::MeasuredStartup,
                    average_ns: u64::try_from(average.as_nanos()).unwrap_or(u64::MAX),
                },
            );
            Self::snapshot(&state)
        };
        self.persist(snapshot);
    }

    pub(in crate::backend) fn abandon_dense(&self, request: DensePlanRequest) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.dense_inflight.remove(&request);
        }
    }
}

pub(super) fn stored_entries(
    entries: &HashMap<DensePlanRequest, DenseRuntimeEntry>,
) -> Vec<StoredDenseEntry> {
    let mut stored = entries
        .iter()
        .map(|(request, entry)| StoredDenseEntry {
            request: *request,
            execution: entry.execution,
            average_ns: entry.average_ns,
        })
        .collect::<Vec<_>>();
    stored.sort_by_key(|entry| {
        (
            entry.request.phase as u8,
            entry.request.role as u8,
            entry.request.tokens,
            entry.request.input_features,
            entry.request.output_features,
        )
    });
    stored
}
