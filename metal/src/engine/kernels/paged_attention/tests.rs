use super::partial_blocks;

#[test]
fn increases_partial_parallelism_for_few_kv_heads() {
    assert_eq!(partial_blocks(8_192, 8, 8), 128);
    assert_eq!(partial_blocks(8_192, 8, 2), 512);
    assert_eq!(partial_blocks(32_768, 8, 2), 1_024);
}
