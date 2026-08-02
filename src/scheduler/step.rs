use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GenerationStepPlan {
    pub(super) decode_tokens: usize,
    pub(super) prefill_tokens: usize,
}

/// Tracks the token budget shared by decode rows and pending prefill work.
pub struct GenerationStepState {
    max_tokens: usize,
    max_decode_rows: usize,
    active_decode_rows: AtomicUsize,
}

impl GenerationStepState {
    pub fn new(max_tokens: usize, max_decode_rows: usize) -> Self {
        let max_tokens = max_tokens.max(1);
        Self {
            max_tokens,
            max_decode_rows: max_decode_rows.max(1).min(max_tokens),
            active_decode_rows: AtomicUsize::new(0),
        }
    }

    pub(super) fn register_decode(&self) {
        self.active_decode_rows.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn release_decode(&self) {
        let mut rows = self.active_decode_rows.load(Ordering::Acquire);
        loop {
            let Some(next) = rows.checked_sub(1) else {
                debug_assert!(false, "released an unregistered decode row");
                return;
            };
            match self.active_decode_rows.compare_exchange_weak(
                rows,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(current) => rows = current,
            }
        }
    }

    pub(super) fn plan(&self) -> GenerationStepPlan {
        let active = self.active_decode_rows.load(Ordering::Acquire);
        let decode_tokens = active.min(self.max_decode_rows).min(self.max_tokens.saturating_sub(1));
        GenerationStepPlan {
            decode_tokens,
            prefill_tokens: self.max_tokens - decode_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationStepPlan, GenerationStepState};

    #[test]
    fn prefill_receives_the_unused_step_budget() {
        let state = GenerationStepState::new(8, 4);
        for _ in 0..3 {
            state.register_decode();
        }
        assert_eq!(state.plan(), GenerationStepPlan { decode_tokens: 3, prefill_tokens: 5 });
        state.release_decode();
        assert_eq!(state.plan().prefill_tokens, 6);
    }

    #[test]
    fn plan_preserves_prefill_progress_at_decode_capacity() {
        let state = GenerationStepState::new(4, 4);
        for _ in 0..8 {
            state.register_decode();
        }
        assert_eq!(state.plan(), GenerationStepPlan { decode_tokens: 3, prefill_tokens: 1 });
    }
}
