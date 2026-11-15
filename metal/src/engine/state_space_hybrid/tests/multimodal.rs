//! Image prefill splices soft tokens over the image span and otherwise runs
//! the plain causal prefill: soft tokens equal to the span's own embeddings
//! reproduce it, and other soft tokens change the continuation.

use super::{assert_close, harness, step_tokens, tokens};
use crate::engine::{Array, ImageTokenSpan, Result};

#[test]
fn spliced_image_embeddings_extend_the_plain_prefill() -> Result<()> {
    let harness = harness(None)?;
    let prompt = tokens(3, 22);
    let (head, tail) = prompt.split_at(prompt.len() - 1);
    let ids = Array::from_u32(head, &[1, i32::try_from(head.len())?])?;
    let span = ImageTokenSpan::new(4..11, head.len())?;
    let own = harness
        .model
        .embedding
        .lookup(&Array::from_u32(&head[4..11], &[1, 7])?, &harness.stream)?;
    let position = i32::try_from(head.len())?;

    let mut plain = harness.model.new_cache(&harness.stream)?;
    harness
        .model
        .forward_prefill(&ids, &mut plain, 0, &harness.stream)?
        .async_eval(&harness.stream)?;
    let expected = step_tokens(&harness, &mut plain, tail, position)?;

    let mut spliced = harness.model.new_cache(&harness.stream)?;
    harness
        .model
        .forward_multimodal_prefill(&ids, &own, span, &mut spliced, &harness.stream)?
        .async_eval(&harness.stream)?;
    assert_eq!(spliced.cached_tokens()?, head.len());
    assert_close(&step_tokens(&harness, &mut spliced, tail, position)?, &expected, 1e-5);

    let other = own.multiply_scalar(-1.0, &harness.stream)?;
    let mut changed = harness.model.new_cache(&harness.stream)?;
    harness
        .model
        .forward_multimodal_prefill(&ids, &other, span, &mut changed, &harness.stream)?
        .async_eval(&harness.stream)?;
    let changed = step_tokens(&harness, &mut changed, tail, position)?;
    assert!(changed.iter().zip(&expected).any(|(left, right)| (left - right).abs() > 1e-3));

    assert!(
        harness
            .model
            .forward_multimodal_prefill(&ids, &own, span, &mut plain, &harness.stream)
            .is_err(),
        "a cache that already holds tokens is refused"
    );
    Ok(())
}
