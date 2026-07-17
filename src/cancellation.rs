use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::{Error, Result};

#[derive(Debug, Clone, Default)]
/// Thread-safe signal used to stop generation between inference steps.
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Creates a token in the non-cancelled state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks this token and all of its clones as cancelled.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_shared_and_idempotent() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(!clone.is_cancelled());
        token.cancel();
        token.cancel();
        assert!(clone.is_cancelled());
        assert!(matches!(clone.check(), Err(Error::Cancelled)));
    }
}
