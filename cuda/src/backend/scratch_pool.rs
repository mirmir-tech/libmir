//! Per-shape scratch shared by every layer of a runtime.
//!
//! Layers execute one after another on the runtime stream, so a plan needs
//! one set of layer-internal buffers per shape, not one per layer: a
//! 1,024-token prefill batch of a 64-layer model otherwise held 8 GB of
//! recurrence and attention scratch that only one layer used at a time.

use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex, MutexGuard, Weak},
};

use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct SharedScratch<S>(Arc<Mutex<S>>);

impl<S> SharedScratch<S> {
    /// Scratch of one plan alone, outside any pool.
    pub(super) fn new(scratch: S) -> Self {
        Self(Arc::new(Mutex::new(scratch)))
    }

    /// A second handle, so a lock can be held while `self` is borrowed mutably.
    pub(super) fn clone_handle(&self) -> Self {
        Self(Arc::clone(&self.0))
    }

    /// Held for the whole execution of one layer; the next layer waits.
    pub(super) fn lock(&self) -> Result<MutexGuard<'_, S>> {
        self.0
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("shared scratch lock is poisoned"))
    }
}

#[derive(Debug)]
pub(super) struct ScratchPool<K, S> {
    entries: Mutex<HashMap<K, Weak<Mutex<S>>>>,
}

impl<K, S> Default for ScratchPool<K, S> {
    fn default() -> Self {
        Self { entries: Mutex::new(HashMap::new()) }
    }
}

impl<K: Clone + Eq + Hash, S> ScratchPool<K, S> {
    /// Returns the scratch registered for `key`, building it when no plan
    /// holds one. Entries live as long as some plan references them.
    pub(super) fn acquire(
        &self,
        key: K,
        build: impl FnOnce() -> Result<S>,
    ) -> Result<SharedScratch<S>> {
        let Ok(mut entries) = self.entries.lock() else {
            return Err(Error::InvalidExecutionPlan("scratch pool lock is poisoned"));
        };
        if let Some(scratch) = entries.get(&key).and_then(Weak::upgrade) {
            return Ok(SharedScratch(scratch));
        }
        entries.retain(|_, entry| entry.strong_count() > 0);
        let scratch = Arc::new(Mutex::new(build()?));
        entries.insert(key, Arc::downgrade(&scratch));
        drop(entries);
        Ok(SharedScratch(scratch))
    }
}
