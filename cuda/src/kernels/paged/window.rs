//! The keys a windowed attention layer's query sees, and the window argument
//! the paged attention kernels take for it.

use models::semantic::AttentionWindow;

use super::super::geometry::narrow;
use crate::{Error, Result};

/// The kernels' marker of a chunk size (`LIBMIR_WINDOW_CHUNKED`).
const CHUNKED: u32 = 1 << 31;

/// The kernel window argument: zero for full attention, a sliding window's
/// token count, or a chunk size with the high bit set.
pub fn argument(window: Option<AttentionWindow>) -> Result<u32> {
    let (tokens, marker) = match window {
        None => return Ok(0),
        Some(AttentionWindow::Sliding(tokens)) => (tokens, 0),
        Some(AttentionWindow::Chunked { chunk }) => (chunk, CHUNKED),
    };
    let tokens = narrow(tokens)?;
    if tokens == 0 || tokens & CHUNKED != 0 {
        return Err(Error::InvalidPagedKv("attention window is empty or too large"));
    }
    Ok(tokens | marker)
}

/// The window of a kernel window argument, as [`argument`] wrote it.
pub fn from_argument(window: u32) -> Result<Option<AttentionWindow>> {
    let tokens = usize::try_from(window & !CHUNKED)?;
    Ok(match (window, window & CHUNKED != 0) {
        (0, _) => None,
        (_, true) => Some(AttentionWindow::Chunked { chunk: tokens }),
        (_, false) => Some(AttentionWindow::Sliding(tokens)),
    })
}

/// The keys the newest of `tokens` positions sees.
#[must_use]
pub const fn visible(tokens: usize, window: Option<AttentionWindow>) -> usize {
    match window {
        Some(window) => window.visible_keys(tokens),
        None => tokens,
    }
}

/// The most keys any query of up to `tokens` positions sees, for sizing
/// work shared by sequences of different lengths.
#[must_use]
pub const fn visible_bound(tokens: usize, window: Option<AttentionWindow>) -> usize {
    match window {
        Some(window) if window.span() < tokens => window.span(),
        _ => tokens,
    }
}

/// The window's token count, sliding or chunked, for tuning keys.
#[must_use]
pub const fn span(window: Option<AttentionWindow>) -> Option<usize> {
    match window {
        Some(window) => Some(window.span()),
        None => None,
    }
}

/// Whether paged `FlashAttention` masks `window`: full and sliding windows,
/// not chunks.
#[must_use]
pub const fn flash_masks(window: Option<AttentionWindow>) -> bool {
    !matches!(window, Some(AttentionWindow::Chunked { .. }))
}

/// Paged `FlashAttention`'s mask of `window`; `None` for a chunked window,
/// which `FlashAttention` cannot mask, so those layers take the paged kernels.
pub fn flash(window: Option<AttentionWindow>) -> Result<Option<mircuda::FmhaCausalWindow>> {
    Ok(match window {
        None => Some(mircuda::FmhaCausalWindow::Full),
        Some(AttentionWindow::Sliding(span)) => Some(mircuda::FmhaCausalWindow::sliding(span)?),
        Some(AttentionWindow::Chunked { .. }) => None,
    })
}

/// [`visible`] of a kernel window argument, as the kernels compute it
/// (`libmir_window_first`); the tests hold it to [`AttentionWindow`].
#[must_use]
pub const fn visible_argument(tokens: u32, window: u32) -> u32 {
    if window == 0 || tokens == 0 {
        tokens
    } else if window & CHUNKED != 0 {
        let chunk = window & !CHUNKED;
        tokens - (tokens - 1) / chunk * chunk
    } else if tokens < window {
        tokens
    } else {
        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_queries_see_their_chunk_and_sliding_ones_their_window() -> Result<()> {
        let chunked = Some(AttentionWindow::Chunked { chunk: 4 });
        let sliding = Some(AttentionWindow::Sliding(4));
        for tokens in 0..12_usize {
            let narrow = u32::try_from(tokens)?;
            assert_eq!(visible(tokens, None), tokens);
            assert_eq!(visible(tokens, sliding), tokens.min(4));
            let expected = if tokens == 0 {
                0
            } else {
                (tokens - 1) % 4 + 1
            };
            assert_eq!(visible(tokens, chunked), expected);
            for window in [None, sliding, chunked] {
                let visible = u32::try_from(visible(tokens, window))?;
                assert_eq!(visible_argument(narrow, argument(window)?), visible);
                assert!(visible_bound(tokens, window) >= visible as usize);
            }
        }
        Ok(())
    }
}
