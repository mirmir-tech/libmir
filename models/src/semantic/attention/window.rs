//! Which keys a query attends to and how strongly by its position.

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Which earlier tokens a query attends to besides itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttentionWindow {
    /// The latest `n` tokens (written as the bare count).
    Sliding(NonZeroUsize),
    /// The tokens since the start of the query's `chunk`-aligned block
    /// (Llama 4's chunked attention).
    Chunked { chunk: NonZeroUsize },
}

impl AttentionWindow {
    /// The sliding window's span, for runtimes that execute only those.
    #[must_use]
    pub const fn sliding(self) -> Option<usize> {
        match self {
            Self::Sliding(span) => Some(span.get()),
            Self::Chunked { .. } => None,
        }
    }

    /// The most keys a query sees: the window, or the chunk.
    #[must_use]
    pub const fn span(self) -> usize {
        match self {
            Self::Sliding(span) | Self::Chunked { chunk: span } => span.get(),
        }
    }

    /// The first key position the query at zero-based `position` sees.
    #[must_use]
    pub const fn first_visible(self, position: usize) -> usize {
        match self {
            Self::Sliding(span) => (position + 1).saturating_sub(span.get()),
            Self::Chunked { chunk } => position / chunk.get() * chunk.get(),
        }
    }

    /// The keys the newest of `tokens` positions sees.
    #[must_use]
    pub const fn visible_keys(self, tokens: usize) -> usize {
        match tokens.checked_sub(1) {
            Some(last) => tokens - self.first_visible(last),
            None => 0,
        }
    }
}

/// Llama 4 attention temperature tuning.
///
/// The query at zero-based position `p` scales by
/// `1 + beta · ln(1 + floor((p + offset) / window))`, which is one until
/// `p + offset` reaches `window`. Mistral 4 counts from zero; Llama 4
/// (transformers `Llama4TextAttention`) from one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryTemperature {
    pub beta: f64,
    pub window: NonZeroUsize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub offset: usize,
}

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde passes the field by reference")]
const fn is_zero(value: &usize) -> bool {
    *value == 0
}

impl QueryTemperature {
    /// A finite `beta`.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.beta.is_finite()
    }

    /// The factor on a query at `position`.
    #[must_use]
    pub fn factor(&self, position: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let periods = ((position + self.offset) / self.window.get()) as f64;
        self.beta.mul_add(periods.ln_1p(), 1.0)
    }

    /// The first position whose factor is not one.
    #[must_use]
    pub const fn first_scaled_position(&self) -> usize {
        self.window.get().saturating_sub(self.offset)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::AttentionWindow;

    /// Llama 4's position 8192 starts the second 8192-token chunk; a sliding
    /// window of 4 always ends at the query.
    #[test]
    fn chunks_restart_at_their_boundary_and_windows_slide() -> Result<(), &'static str> {
        let tokens = |count| NonZeroUsize::new(count).ok_or("empty window");
        let chunked = AttentionWindow::Chunked { chunk: tokens(8192)? };
        assert_eq!(chunked.first_visible(8191), 0);
        assert_eq!(chunked.first_visible(8192), 8192);
        assert_eq!(chunked.visible_keys(8193), 1);
        assert_eq!(chunked.visible_keys(16_384), 8192);
        let sliding = AttentionWindow::Sliding(tokens(4)?);
        assert_eq!(sliding.first_visible(2), 0);
        assert_eq!(sliding.first_visible(9), 6);
        assert_eq!(sliding.visible_keys(10), 4);
        assert_eq!(sliding.visible_keys(0), 0);
        Ok(())
    }

    /// A window of no tokens is not a window: specs that declare one fail to
    /// load rather than reaching a backend.
    #[test]
    fn empty_windows_do_not_deserialize() {
        for empty in [serde_json::json!(0), serde_json::json!({ "chunk": 0 })] {
            assert!(serde_json::from_value::<AttentionWindow>(empty).is_err());
        }
        assert_eq!(
            serde_json::from_value::<AttentionWindow>(serde_json::json!({ "chunk": 2 })).ok(),
            NonZeroUsize::new(2).map(|chunk| AttentionWindow::Chunked { chunk })
        );
    }
}
