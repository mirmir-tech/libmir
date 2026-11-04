use serde::{Deserialize, Serialize};

/// Bounded speculative window. Each draft retains a recurrent-state snapshot;
/// the operational ceiling prevents accidental unbounded rollback allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct DraftTokens(usize);

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("speculative draft tokens must be between 1 and 8")]
pub struct InvalidDraftTokens;

impl DraftTokens {
    pub const MAX: Self = Self(8);
    pub const MIN: Self = Self(1);

    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for DraftTokens {
    type Error = InvalidDraftTokens;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if (Self::MIN.0..=Self::MAX.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(InvalidDraftTokens)
        }
    }
}

impl From<DraftTokens> for usize {
    fn from(value: DraftTokens) -> Self {
        value.get()
    }
}

impl std::fmt::Display for DraftTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_the_rollback_window() {
        for value in [0, 9, 1_000_000, usize::MAX] {
            assert!(DraftTokens::try_from(value).is_err());
        }
        for value in 1..=8 {
            assert_eq!(DraftTokens::try_from(value).map(DraftTokens::get).ok(), Some(value));
        }
    }
}
