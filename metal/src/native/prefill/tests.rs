use super::required_prefill_pages;

#[test]
fn reserves_copy_on_write_page_for_partial_prefix_tail() {
    assert_eq!(required_prefill_pages(6_145, 4_098, 16), 130);
    assert_eq!(required_prefill_pages(6_145, 4_096, 16), 130);
    assert_eq!(required_prefill_pages(6_145, 0, 16), 386);
}

#[test]
fn exact_prefix_retains_decode_and_copy_on_write_headroom() {
    assert_eq!(required_prefill_pages(4_098, 4_098, 16), 2);
    assert_eq!(required_prefill_pages(4_096, 4_096, 16), 1);
}
