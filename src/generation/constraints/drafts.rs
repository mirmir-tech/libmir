//! Token masks for speculative drafts under the tool grammar.

use llguidance::Matcher;
use runtime::{backend::TokenMask, speculation::MaskedDraft};

use super::{invalid, mask_words};
use crate::Result;

/// The leading `drafts` the grammar allows, each with the mask of the token
/// after it. A draft that ends the grammar is left for the step to choose
/// again. The matcher keeps its state.
pub(super) fn masks(
    matcher: &mut Matcher,
    drafts: &[u32],
    vocab: usize,
) -> Result<Vec<MaskedDraft>> {
    let valid = matcher.validate_tokens(drafts).map_err(|error| invalid(error.to_string()))?;
    let mut masked = Vec::with_capacity(valid);
    let mut consumed = 0;
    let result = (|| {
        for &token in &drafts[..valid] {
            matcher.consume_token(token).map_err(|error| invalid(error.to_string()))?;
            consumed += 1;
            if matcher.is_stopped() {
                break;
            }
            let bits = matcher.compute_mask().map_err(|error| invalid(error.to_string()))?;
            if bits.is_zero() {
                break;
            }
            let mask = TokenMask::new(mask_words(&bits, vocab)?, vocab)?;
            masked.push(MaskedDraft { token, mask });
        }
        Ok(())
    })();
    matcher.rollback(consumed).map_err(|error| invalid(error.to_string()))?;
    result.map(|()| masked)
}
