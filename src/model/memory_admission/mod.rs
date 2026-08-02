use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use super::memory_policy;
use crate::{Error, MemoryRuntimeConfig, MemorySnapshot, ModelMemoryEstimate, Result};

mod lease;
#[cfg(test)]
mod tests;

pub(super) use lease::ModelMemoryLease;

#[derive(Clone, Debug, Default)]
pub(super) struct ModelMemoryManager {
    loads: Arc<Mutex<()>>,
    ledger: Arc<Mutex<MemoryLedger>>,
}

#[derive(Debug, Default)]
struct MemoryLedger {
    next_id: u64,
    entries: HashMap<u64, Reservation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Reservation {
    model: String,
    bytes: u64,
    state: ReservationState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReservationState {
    Loading,
    Resident,
}

impl ModelMemoryManager {
    pub(super) fn serialize_load(&self) -> Result<MutexGuard<'_, ()>> {
        let Ok(guard) = self.loads.lock() else {
            return Err(poisoned("model load gate"));
        };
        Ok(guard)
    }

    pub(super) fn reserve(
        &self,
        model: String,
        estimate: ModelMemoryEstimate,
        memory: &MemorySnapshot,
        policy: MemoryRuntimeConfig,
        allow_overcommit: bool,
    ) -> Result<ModelMemoryLease> {
        let Ok(mut ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        let committed = ledger
            .entries
            .values()
            .map(|entry| entry.bytes)
            .fold(0_u64, u64::saturating_add);
        let planned = memory_policy::planned_residency(estimate, memory);
        if !allow_overcommit
            && let Some(available) = available_budget(memory, policy, committed)
            && planned > available
        {
            return Err(Error::MemoryAdmission {
                model,
                required_bytes: planned,
                available_bytes: available,
            });
        }
        let id = ledger.next_id;
        ledger.next_id = ledger.next_id.wrapping_add(1);
        ledger.entries.insert(
            id,
            Reservation {
                model: model.clone(),
                bytes: planned,
                state: ReservationState::Loading,
            },
        );
        tracing::info!(
            model,
            reservation_bytes = planned,
            committed_bytes = committed.saturating_add(planned),
            "reserved accelerator memory for model load"
        );
        drop(ledger);
        Ok(ModelMemoryLease::new(id, self.ledger.clone()))
    }

    pub(super) fn committed_bytes(&self) -> Result<u64> {
        let Ok(ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        Ok(ledger
            .entries
            .values()
            .map(|entry| entry.bytes)
            .fold(0_u64, u64::saturating_add))
    }
}

fn available_budget(
    memory: &MemorySnapshot,
    policy: MemoryRuntimeConfig,
    committed: u64,
) -> Option<u64> {
    let available = memory.available_bytes?.saturating_add(memory.cached_bytes);
    let reserve = memory_policy::platform_reserve(policy, memory);
    let physical = available.saturating_sub(reserve);
    let logical = memory
        .total_bytes
        .map(|total| total.saturating_sub(reserve).saturating_sub(committed));
    Some(logical.map_or(physical, |logical| physical.min(logical)))
}

fn poisoned(target: &str) -> Error {
    runtime::RuntimeError::Config(format!("{target} is poisoned")).into()
}
