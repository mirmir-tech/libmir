use super::physical_page_capacity;

#[test]
fn physical_capacity_covers_session_fragmentation_reservation_and_cow() {
    assert_eq!(physical_page_capacity(6_177, 16, 16), 6_240);
    assert_eq!(physical_page_capacity(6_177, 0, 16), 6_192);
}

#[test]
fn physical_capacity_covers_observed_fragmented_shared_batch_tail() {
    assert_eq!(physical_page_capacity(6_177, 16, 16), 6_240);
}
