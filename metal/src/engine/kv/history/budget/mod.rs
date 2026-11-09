use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Admission {
    Open,
    Suspended,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Snapshot {
    pub limit_bytes: usize,
    pub retained_bytes: usize,
    pub admission: Admission,
    /// Set by an explicit limit; automatic memory sizing then leaves it alone.
    pub pinned_limit: bool,
}

#[derive(Debug)]
pub struct Budget(Mutex<Snapshot>);

impl Default for Budget {
    fn default() -> Self {
        Self::new(512 * 1024 * 1024)
    }
}

impl Budget {
    pub fn new(limit_bytes: usize) -> Self {
        Self(Mutex::new(Snapshot {
            limit_bytes,
            retained_bytes: 0,
            admission: Admission::Open,
            pinned_limit: false,
        }))
    }

    pub fn snapshot(&self) -> Snapshot {
        *self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    pub fn set_limit(&self, limit_bytes: usize) {
        let mut state = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.limit_bytes = limit_bytes;
        state.pinned_limit = true;
    }

    /// Follows live memory headroom unless a caller pinned the limit.
    pub fn set_automatic_limit(&self, limit_bytes: usize) {
        let mut state = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.pinned_limit {
            state.limit_bytes = limit_bytes;
        }
    }

    pub fn admits(&self, bytes: usize) -> bool {
        let state = self.snapshot();
        state.admission == Admission::Open
            && state.retained_bytes.checked_add(bytes).is_some_and(|t| t <= state.limit_bytes)
    }

    pub fn set_admission(&self, admission: Admission) {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).admission = admission;
    }

    pub fn allows_reuse(&self) -> bool {
        let state = self.snapshot();
        state.admission == Admission::Open && state.retained_bytes <= state.limit_bytes
    }

    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Arc<Lease>> {
        let mut state = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let total = state.retained_bytes.checked_add(bytes)?;
        if state.admission == Admission::Suspended || total > state.limit_bytes {
            return None;
        }
        state.retained_bytes = total;
        drop(state);
        Some(Arc::new(Lease { budget: Arc::clone(self), bytes }))
    }
}

// Counts storage owned by retained History objects. Lazy graph readers and
// allocator caches have separate lifetimes; this is not a total MLX memory cap.
#[derive(Debug)]
pub struct Lease {
    budget: Arc<Budget>,
    bytes: usize,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.budget
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retained_bytes -= self.bytes;
    }
}

pub fn storage_bytes(shape: [usize; 4], dtype: mirtal::DType) -> crate::engine::Result<usize> {
    use crate::engine::Error;
    let pair_bytes = match dtype {
        mirtal::DType::Float32 => 8,
        mirtal::DType::Float16 | mirtal::DType::Bfloat16 => 4,
        _ => return Err(Error::InvalidModel("unsupported persistent history dtype".into())),
    };
    shape
        .into_iter()
        .try_fold(pair_bytes, usize::checked_mul)
        .ok_or(Error::ShapeOverflow)
}
