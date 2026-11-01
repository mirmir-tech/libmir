//! Speculative decoding with a checkpoint's multi-token prediction modules.

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Each decode step drafts `draft_tokens` tokens with the prediction module
/// and verifies them, with the token after them, in one target pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Speculation {
    pub draft_tokens: NonZeroUsize,
}

/// What one speculative decode step may keep after the fed token. A host that
/// constrains or ends generation at some tokens bounds acceptance by them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftBounds {
    /// Keeps at most this many drafts; `None` leaves acceptance unbounded.
    pub max_accepted: Option<usize>,
    /// Acceptance stops at the first draft among these, which becomes the
    /// step's token, so the host sees it before any token after it.
    pub stop_tokens: Vec<u32>,
}

impl DraftBounds {
    /// Bounds `accepted` of `drafts`, kept with `next` after them. A shorter
    /// prefix ends at a kept draft, which is a token the verification chose.
    #[must_use]
    pub fn apply(&self, drafts: &[u32], accepted: usize, next: u32) -> (usize, u32) {
        let stop = drafts[..accepted].iter().position(|draft| self.stop_tokens.contains(draft));
        let kept = stop.unwrap_or(accepted).min(self.max_accepted.unwrap_or(usize::MAX));
        if kept == accepted {
            (accepted, next)
        } else {
            (kept, drafts[kept])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stop_draft_becomes_the_step_token() {
        let bounds = DraftBounds { max_accepted: None, stop_tokens: vec![9] };
        assert_eq!(bounds.apply(&[4, 9, 5], 3, 6), (1, 9));
        assert_eq!(bounds.apply(&[4, 5, 9], 2, 9), (2, 9));
    }

    #[test]
    fn the_cap_ends_at_a_kept_draft() {
        let bounds = DraftBounds {
            max_accepted: Some(1),
            stop_tokens: Vec::new(),
        };
        assert_eq!(bounds.apply(&[4, 5, 6], 3, 7), (1, 5));
        assert_eq!(bounds.apply(&[4, 5, 6], 0, 7), (0, 7));
    }
}
