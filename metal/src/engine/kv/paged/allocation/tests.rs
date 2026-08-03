use super::growth_target;

#[test]
fn arena_growth_is_geometric_and_bounded() -> crate::engine::Result<()> {
    assert_eq!(growth_target(0, 6_177, 32, 6_177)?, 6_177);
    assert_eq!(growth_target(32, 33, 32, 6_177)?, 64);
    assert_eq!(growth_target(64, 65, 32, 6_177)?, 128);
    assert_eq!(growth_target(4_096, 4_097, 32, 6_177)?, 6_177);
    Ok(())
}

#[test]
fn arena_growth_rejects_the_logical_limit() {
    assert!(growth_target(4_096, 6_178, 32, 6_177).is_err());
}
