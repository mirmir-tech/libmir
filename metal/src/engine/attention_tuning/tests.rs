use super::{
    PagedExecution, candidates,
    profile::{AttentionKey, context_bucket, fallback},
};

fn key(bucket: usize) -> AttentionKey {
    AttentionKey {
        context_bucket: bucket,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 256,
        page_size: 16,
        dtype: 5,
    }
}

#[test]
fn groups_growing_contexts_into_bounded_profiles() {
    assert_eq!(context_bucket(1_024), 1_024);
    assert_eq!(context_bucket(1_025), 2_048);
    assert_eq!(context_bucket(8_191), 8_192);
    assert_eq!(context_bucket(8_192), 8_192);
    assert_eq!(context_bucket(32_769), 65_536);
    assert_eq!(context_bucket(100_000), 131_072);
}

#[test]
fn supported_profile_retains_the_existing_two_pass_fallback() {
    let expected = PagedExecution::TwoPass { blocks: 64, reduction_groups: 32 };
    assert_eq!(fallback(key(1_024), 1_024), expected);
    assert_eq!(fallback(key(8_192), 8_192), expected);
}

#[test]
fn tunes_bounded_geometry_around_the_heuristic() {
    let choices = candidates(PagedExecution::TwoPass { blocks: 128, reduction_groups: 32 });
    assert_eq!(choices.len(), 10);
    assert_eq!(choices[0], PagedExecution::Direct);
    for blocks in [64, 128, 256] {
        for reduction_groups in [8, 16, 32] {
            assert!(choices.contains(&PagedExecution::TwoPass { blocks, reduction_groups }));
        }
    }
}
