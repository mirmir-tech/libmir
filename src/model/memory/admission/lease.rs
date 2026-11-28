use std::sync::{Arc, Mutex};

use super::{MemoryLedger, ReservationState, poisoned};
use crate::Result;

#[derive(Debug)]
pub(in crate::model) struct ModelMemoryLease {
    id: u64,
    ledger: Arc<Mutex<MemoryLedger>>,
}

impl ModelMemoryLease {
    pub(super) const fn new(id: u64, ledger: Arc<Mutex<MemoryLedger>>) -> Self {
        Self { id, ledger }
    }

    pub(in crate::model) fn mark_resident(&self) -> Result<()> {
        let Ok(mut ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        let reservation = ledger.entries.get_mut(&self.id).ok_or_else(|| {
            runtime::RuntimeError::Config("model memory reservation is missing".into())
        })?;
        reservation.state = ReservationState::Resident;
        let model = reservation.model.clone();
        let bytes = reservation.bytes;
        drop(ledger);
        tracing::info!(
            %model,
            reservation_bytes = bytes,
            "committed resident model memory reservation"
        );
        Ok(())
    }
}

impl ModelMemoryLease {
    /// Records the residency of a model whose K/V cache was resized.
    /// Bytes every other reservation in the ledger holds: what other loaded
    /// models take beside this one.
    pub(in crate::model) fn others_bytes(&self) -> Result<u64> {
        let Ok(ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        Ok(ledger.others_bytes(self.id))
    }

    pub(in crate::model) fn resize(&self, estimate: crate::ModelMemoryEstimate) -> Result<()> {
        let Ok(mut ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        let reservation = ledger.entries.get_mut(&self.id).ok_or_else(|| {
            runtime::RuntimeError::Config("model memory reservation is missing".into())
        })?;
        // CUDA models never share pages, so the estimate is the residency.
        reservation.bytes = estimate.required_bytes;
        Ok(())
    }
}

impl ModelMemoryLease {
    /// Bytes this model holds that its unload would give back: its own
    /// reservation, and a shared K/V reservation no other model uses.
    pub(in crate::model) fn released_bytes(&self) -> Result<u64> {
        let Ok(ledger) = self.ledger.lock() else {
            return Err(poisoned("model memory ledger"));
        };
        let Some(reservation) = ledger.entries.get(&self.id) else {
            return Ok(0);
        };
        let shared = reservation
            .shared_memory
            .and_then(|id| ledger.shared.get(&id))
            .filter(|shared| shared.references == 1)
            .map_or(0, |shared| shared.bytes);
        Ok(reservation.bytes.saturating_add(shared))
    }
}

impl Drop for ModelMemoryLease {
    fn drop(&mut self) {
        let Ok(mut ledger) = self.ledger.lock() else {
            return;
        };
        if let Some(reservation) = ledger.entries.remove(&self.id) {
            if let Some(shared) = reservation.shared_memory
                && let Some(shared_reservation) = ledger.shared.get_mut(&shared)
            {
                shared_reservation.references = shared_reservation.references.saturating_sub(1);
                let released = shared_reservation.references == 0;
                if released {
                    ledger.shared.remove(&shared);
                }
            }
            tracing::info!(
                model = %reservation.model,
                reservation_bytes = reservation.bytes,
                state = ?reservation.state,
                "released model memory reservation"
            );
        }
    }
}
