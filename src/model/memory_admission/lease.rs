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
            model,
            reservation_bytes = bytes,
            "committed resident model memory reservation"
        );
        Ok(())
    }
}

impl Drop for ModelMemoryLease {
    fn drop(&mut self) {
        let Ok(mut ledger) = self.ledger.lock() else {
            return;
        };
        if let Some(reservation) = ledger.entries.remove(&self.id) {
            tracing::info!(
                model = reservation.model,
                reservation_bytes = reservation.bytes,
                state = ?reservation.state,
                "released model memory reservation"
            );
        }
    }
}
