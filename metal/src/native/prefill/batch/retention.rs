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
