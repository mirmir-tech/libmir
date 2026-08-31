use super::plan::{
    checkpoint_distance, context_chunk_budget, fair_chunk_budget, reusable_prefix_tokens,
    round_rows_from_pending, row_chunk_budget, valid_chunk,
};

#[test]
fn reuses_every_complete_cached_block_before_the_missing_suffix() {
    assert_eq!(reusable_prefix_tokens(4_080, 6_144, Some(16), 0), 4_080);
}

#[test]
fn reprocesses_the_last_block_when_the_complete_prompt_is_cached() {
    assert_eq!(reusable_prefix_tokens(4_096, 4_096, Some(16), 0), 4_080);
}

#[test]
fn rejects_invalid_or_unallocated_prefix_ranges() {
    assert_eq!(reusable_prefix_tokens(8_192, 4_096, None, 0), 0);
    assert_eq!(reusable_prefix_tokens(0, 4_096, Some(16), 0), 0);
}

#[test]
fn windowed_prefix_reuse_replays_the_full_layer_receptive_field() {
    assert_eq!(reusable_prefix_tokens(8_192, 10_240, Some(16), 1_525), 6_667);
    assert_eq!(reusable_prefix_tokens(1_024, 4_096, Some(16), 1_525), 0);
}

#[test]
fn rotates_each_round_and_skips_completed_rows() {
    assert_eq!(round_rows_from_pending(&[true, false, true, true], 2), vec![2, 3, 0]);
    assert_eq!(round_rows_from_pending(&[true, false, true, true], 3), vec![3, 0, 2]);
}

#[test]
fn rejects_chunks_that_exceed_the_step_budget() {
    assert!(valid_chunk(2_048, 32_768, 8_192));
    assert!(!valid_chunk(8_193, 32_768, 8_192));
    assert!(!valid_chunk(2_049, 2_048, 8_192));
    assert!(!valid_chunk(0, 32_768, 8_192));
}

#[test]
fn shares_the_remaining_budget_across_every_pending_row() {
    assert_eq!(fair_chunk_budget(8_192, 10), 820);
    assert_eq!(fair_chunk_budget(7_680, 9), 854);
    assert_eq!(fair_chunk_budget(1_024, 1), 1_024);
}

#[test]
fn reduces_query_chunks_after_attention_context_becomes_expensive() {
    assert_eq!(context_chunk_budget(0, 4, 8_192, false, false, false), usize::MAX);
    assert_eq!(context_chunk_budget(2_047, 4, 8_192, false, false, false), usize::MAX);
    assert_eq!(context_chunk_budget(2_048, 4, 8_192, false, false, false), 2_048);
    assert_eq!(context_chunk_budget(32_768, 4, 8_192, false, false, false), 2_048);
    assert_eq!(context_chunk_budget(2_048, 4, 8_191, false, false, false), 1_024);
    assert_eq!(context_chunk_budget(2_048, 4, 8_192, true, false, false), 1_024);
}

#[test]
fn only_checkpoint_restores_bypass_the_interleaved_tail_cap() {
    assert_eq!(context_chunk_budget(6_651, 2, 8_192, false, true, false), usize::MAX);
    assert_eq!(context_chunk_budget(6_651, 2, 8_192, true, true, false), 1_024);
    assert_eq!(context_chunk_budget(6_651, 2, 8_192, true, true, true), usize::MAX);
}

#[test]
fn checkpoint_tails_consume_the_interleaved_budget_completion_first() {
    assert_eq!(row_chunk_budget(8_188, 4, true), 8_188);
    assert_eq!(row_chunk_budget(6_124, 3, true), 6_124);
    assert_eq!(row_chunk_budget(4_060, 2, true), 4_060);
    assert_eq!(row_chunk_budget(1_996, 1, true), 1_996);
    assert_eq!(row_chunk_budget(8_188, 4, false), 2_047);
}

#[test]
fn skips_declared_checkpoints_that_the_backend_cannot_store() {
    assert_eq!(checkpoint_distance(64, &[8_188], Some(8_176), Some(16)), 8_112);
    assert_eq!(checkpoint_distance(8_176, &[8_188], Some(10_224), Some(16)), 2_048);
    assert_eq!(checkpoint_distance(64, &[8_188], None, None), 8_124);
}
