//! A prompt snapshot restored onto the next request's runtime blocks, as
//! the server's prefill does for a repeated prompt.

use runtime::kv::BlockId;

use super::{Harness, harness, step_tokens, tokens};
use crate::engine::{Array, DecoderCache, Result};

#[test]
fn a_snapshot_rebound_to_the_next_requests_blocks_decodes() -> Result<()> {
    let harness = harness(None)?;
    let prompt = tokens(4, 16);
    let blocks = [BlockId(0), BlockId(1)];
    let mut cache = harness.model.new_cache(&harness.stream)?;
    prefill(&harness, &mut cache, &prompt, &blocks)?;
    let snapshot = cache.snapshot_at(16)?;
    let continuation = tokens(5, 2);
    let original = step_tokens(&harness, &mut cache, &continuation, 16)?;
    drop(cache);

    // Rebinding appends the second block to the snapshot's page table, which
    // materializing the recurrent state must leave to the next attention read.
    let mut restored = snapshot.snapshot_at(16)?;
    restored.bind_blocks(&blocks, None, &harness.stream)?;
    let mut roots = Vec::new();
    restored.extend_graph_roots(&mut roots);
    harness.stream.eval_many_with_paged_arenas(&roots)?;
    restored.detach_evaluated_graphs(&harness.stream)?;
    let continued = step_tokens(&harness, &mut restored, &continuation, 16)?;
    super::assert_close(&continued, &original, 1e-5);
    Ok(())
}

/// A prompt ending inside a page resumes from a snapshot keeping a copy of
/// that page, after another request overwrote the block it lived in.
#[test]
fn a_snapshot_restores_its_partial_page_into_a_new_block() -> Result<()> {
    let harness = harness(None)?;
    let prompt = tokens(4, 20);
    let mut cache = harness.model.new_cache(&harness.stream)?;
    prefill(&harness, &mut cache, &prompt, &[BlockId(0), BlockId(1)])?;
    let mut snapshot = cache.snapshot_at(20)?;
    snapshot.capture_tails(&harness.stream)?;
    let continuation = tokens(5, 2);
    let original = step_tokens(&harness, &mut cache, &continuation, 20)?;
    drop(cache);

    let mut other = harness.model.new_cache(&harness.stream)?;
    prefill(&harness, &mut other, &tokens(9, 20), &[BlockId(1), BlockId(2)])?;
    drop(other);

    let mut restored = snapshot.snapshot_at(20)?;
    restored.bind_blocks(&[BlockId(0), BlockId(3)], None, &harness.stream)?;
    restored.restore_tails(&harness.stream)?;
    let continued = step_tokens(&harness, &mut restored, &continuation, 20)?;
    super::assert_close(&continued, &original, 1e-5);
    Ok(())
}

fn prefill(
    harness: &Harness,
    cache: &mut DecoderCache,
    prompt: &[u32],
    blocks: &[BlockId],
) -> Result<()> {
    cache.bind_blocks(blocks, None, &harness.stream)?;
    let hidden = harness.model.forward_prefill(
        &Array::from_u32(prompt, &[1, i32::try_from(prompt.len())?])?,
        cache,
        0,
        &harness.stream,
    )?;
    let mut roots = vec![&hidden];
    cache.extend_prefill_graph_roots(&mut roots);
    harness.stream.eval_many_with_paged_arenas(&roots)?;
    cache.detach_evaluated_graphs(&harness.stream)
}
