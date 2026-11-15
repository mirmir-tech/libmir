use std::sync::{Arc, Mutex, TryLockError, Weak};

use super::{Batch, Result};

#[derive(Debug, Default)]
pub(in crate::native) struct Reservations {
    batches: Vec<Weak<Mutex<Option<Batch>>>>,
}

impl Reservations {
    pub(super) fn register(&mut self, batch: &Arc<Mutex<Option<Batch>>>) {
        self.batches.retain(|batch| batch.strong_count() > 0);
        self.batches.push(Arc::downgrade(batch));
    }

    /// Whether a registered batch still has sequences to prefill.
    pub(in crate::native) fn in_flight(&mut self) -> bool {
        self.batches.retain(|batch| batch.strong_count() > 0);
        self.batches
            .iter()
            .filter_map(Weak::upgrade)
            // A batch locked by its own execution is in flight too.
            .any(|batch| batch.try_lock().map_or(true, |guard| guard.is_some()))
    }

    pub(in crate::native) fn reclaim(&mut self) -> Result<()> {
        self.batches.retain(|batch| batch.strong_count() > 0);
        for weak in &self.batches {
            let Some(batch) = weak.upgrade() else {
                continue;
            };
            let mut guard = match batch.try_lock() {
                Ok(guard) => guard,
                // The executing batch holds its lock and reclaims its own tails
                // before admission. Other backend operations run on this worker.
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Poisoned(error)) => return Err(error.into()),
            };
            if let Some(batch) = guard.as_mut() {
                for sequence in &mut batch.sequences {
                    sequence.reclaim_reservation()?;
                }
            }
        }
        Ok(())
    }
}
