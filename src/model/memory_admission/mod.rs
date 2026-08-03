use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use super::{cache::SharedCacheMemory, memory_policy};
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
    shared: HashMap<u64, SharedReservation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Reservation {
    model: String,
    bytes: u64,
    shared_memory: Option<u64>,
    state: ReservationState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SharedReservation {
    bytes: u64,
    references: usize,
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
        shared_memory: Option<SharedCacheMemory>,
        memory: &MemorySnapshot,
        policy: MemoryRuntimeConfig,
        allow_overcommit: bool,
    ) -> Result<ModelMemoryLease> {
        let Ok(mut ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        let committed = committed(&ledger);
        let shared_bytes =
            shared_memory.map_or(0, |shared| shared.bytes.min(estimate.kv_cache_bytes));
        let model_bytes =
            memory_policy::planned_residency(estimate, memory).saturating_sub(shared_bytes);
        let new_shared_bytes = match shared_memory {
            Some(shared) if !ledger.shared.contains_key(&shared.id) => shared_bytes,
            Some(_) | None => 0,
        };
        let planned = model_bytes.saturating_add(new_shared_bytes);
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
        if let Some(shared) = shared_memory
            && let Some(reservation) = ledger.shared.get(&shared.id)
            && reservation.bytes != shared_bytes
        {
            return Err(runtime::RuntimeError::Config(
                "shared K/V memory reservation changed size".into(),
            )
            .into());
        }
        let id = ledger.next_id;
        ledger.next_id = ledger.next_id.wrapping_add(1);
        ledger.entries.insert(
            id,
            Reservation {
                model: model.clone(),
                bytes: model_bytes,
                shared_memory: shared_memory.map(|shared| shared.id),
                state: ReservationState::Loading,
            },
        );
        if let Some(shared) = shared_memory {
            let reservation = ledger
                .shared
                .entry(shared.id)
                .or_insert(SharedReservation { bytes: shared_bytes, references: 0 });
            reservation.references = reservation.references.saturating_add(1);
        }
        tracing::info!(
            model,
            reservation_bytes = model_bytes,
            shared_cache_bytes = shared_bytes,
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
        Ok(committed(&ledger))
    }
}

fn committed(ledger: &MemoryLedger) -> u64 {
    ledger
        .entries
        .values()
        .map(|entry| entry.bytes)
        .chain(ledger.shared.values().map(|entry| entry.bytes))
        .fold(0_u64, u64::saturating_add)
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
