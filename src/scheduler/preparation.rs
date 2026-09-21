use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Requests between the start of prompt preparation and the end of their
/// prefill. The generation worker waits for companions only while some request
/// it has not yet received is still being prepared.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Clone, Debug, Default)]
pub struct PreparingRequests(Arc<AtomicUsize>);

/// Keeps one request counted until it is dropped.
#[derive(Debug)]
pub struct PreparationGuard(Arc<AtomicUsize>);

#[cfg(any(feature = "cuda", feature = "metal"))]
impl PreparingRequests {
    pub fn announce(&self) -> PreparationGuard {
        self.0.fetch_add(1, Ordering::AcqRel);
        PreparationGuard(Arc::clone(&self.0))
    }

    pub fn count(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }
}

impl Drop for PreparationGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "metal")))]
mod tests {
    use super::PreparingRequests;

    #[test]
    fn guards_count_requests_until_dropped() {
        let preparing = PreparingRequests::default();
        let first = preparing.announce();
        let shared = preparing.clone();
        let second = shared.announce();
        assert_eq!(preparing.count(), 2);
        drop(first);
        assert_eq!(preparing.count(), 1);
        drop(second);
        assert_eq!(preparing.count(), 0);
    }
}
