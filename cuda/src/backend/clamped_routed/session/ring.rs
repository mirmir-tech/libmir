use std::collections::HashMap;

use uuid::Uuid;

use crate::{Error, Result};

#[derive(Debug)]
pub(super) struct SessionRings {
    assigned: HashMap<Uuid, usize>,
    free: Vec<usize>,
}

impl SessionRings {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            assigned: HashMap::with_capacity(capacity),
            free: (0..capacity).rev().collect(),
        }
    }

    pub(super) fn acquire(&mut self, session: Uuid) -> Result<usize> {
        if let Some(slot) = self.assigned.get(&session) {
            return Ok(*slot);
        }
        let Some(slot) = self.free.pop() else {
            return Err(Error::InvalidPagedKv(
                "windowed KV session capacity exceeds max_batch_requests",
            ));
        };
        self.assigned.insert(session, slot);
        Ok(slot)
    }

    pub(super) fn acquire_many(&mut self, sessions: &[Uuid]) -> Result<Vec<usize>> {
        let needed = sessions.iter().filter(|session| !self.assigned.contains_key(session)).count();
        if needed > self.free.len() {
            return Err(Error::InvalidPagedKv(
                "windowed KV session capacity exceeds max_batch_requests",
            ));
        }
        sessions.iter().map(|session| self.acquire(*session)).collect()
    }

    pub(super) fn slot(&self, session: Uuid) -> Result<usize> {
        self.assigned
            .get(&session)
            .copied()
            .ok_or_else(|| Error::State("windowed KV session has no ring slot".into()))
    }

    pub(super) fn release(&mut self, session: Uuid) {
        if let Some(slot) = self.assigned.remove(&session) {
            self.free.push(slot);
        }
    }

    pub(super) fn clear(&mut self) {
        let capacity = self.assigned.len() + self.free.len();
        self.assigned.clear();
        self.free = (0..capacity).rev().collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_stable_and_reused_after_release() -> Result<()> {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut rings = SessionRings::new(1);

        assert_eq!(rings.acquire(first)?, 0);
        assert_eq!(rings.acquire(first)?, 0);
        assert!(rings.acquire(second).is_err());
        rings.release(first);
        assert_eq!(rings.acquire(second)?, 0);
        Ok(())
    }

    #[test]
    fn batch_acquisition_is_atomic_when_capacity_is_exhausted() -> Result<()> {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut rings = SessionRings::new(1);

        assert!(rings.acquire_many(&[first, second]).is_err());
        assert_eq!(rings.acquire(second)?, 0);
        Ok(())
    }
}
