use super::*;

#[test]
fn bounds_automatic_prefix_cache_to_two_fifths_of_usable_memory() {
    let memory = MemoryStats {
        active: 10,
        cached: 20,
        peak: 30,
        limit: 1_000,
        recommended: Some(800),
    };
    assert_eq!(prefix_cache_budget(memory, None), 320);
    assert_eq!(prefix_cache_budget(memory, Some(75)), 75);
}
