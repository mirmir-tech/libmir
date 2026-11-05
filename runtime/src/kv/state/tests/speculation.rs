//! Block reservation and processed tokens of speculative decode steps.

use super::*;

#[test]
fn reserved_draft_blocks_hold_no_tokens_and_commit_only_full_ones() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 8,
        dtype: KvCacheDType::Auto,
    });
    let mut state = KvSessionState::new(Uuid::new_v4(), "nemotron", 2);
    state.prepare_prefill_in_place(&mut cache, &[1, 2])?;
    state.append_decode_in_place(&mut cache, 3)?;
    // The fed token, three drafts and three more after them.
    state.reserve_capacity_in_place(&mut cache, state.token_len() + 7)?;
    assert_eq!(state.token_len(), 3);
    assert_eq!(state.table().blocks().len(), 5);
    assert_eq!(state.table().token_len(), 3);

    state.append_processed_in_place(&mut cache, &[4, 5])?;
    assert_eq!(state.token_len(), 5);
    assert_eq!(state.table().blocks().len(), 5);
    // Blocks of tokens 1–2 and 3–4; token 5's block is still open.
    assert_eq!(state.commit_ready_prefix_blocks(&mut cache)?, 2);
    // The two committed blocks stay cached as a prefix; the rest are freed.
    assert_eq!(state.release(&mut cache)?, 3);
    Ok(())
}
