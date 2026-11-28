use std::sync::{Arc, Mutex, Weak};

use super::Batch;

#[derive(Debug, Default)]
pub(in crate::native) struct ActivePrefills {
    batches: Vec<Weak<Mutex<Option<Batch>>>>,
}

impl ActivePrefills {
    pub(super) fn register(&mut self, batch: &Arc<Mutex<Option<Batch>>>) {
        self.batches.retain(|batch| batch.strong_count() > 0);
        self.batches.push(Arc::downgrade(batch));
    }

    /// Whether any registered prefill batch is still in flight.
    pub(in crate::native) fn in_flight(&mut self) -> bool {
        self.batches.retain(|batch| batch.strong_count() > 0);
        self.batches
            .iter()
            .filter_map(Weak::upgrade)
            // A batch locked by its own execution is in flight too.
            .any(|batch| batch.try_lock().map_or(true, |guard| guard.is_some()))
    }
}
