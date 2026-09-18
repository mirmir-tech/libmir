use super::{HashSet, refill_rows};

#[test]
fn bounds_arrivals_to_spare_slots_and_reserved_tokens() {
    let requests = || [128; 5].into_iter().map(|tokens| (tokens, &[][..]));
    assert_eq!(refill_rows(requests(), 4, 512, HashSet::new(), 100), 4);
    assert_eq!(refill_rows(requests(), 2, 512, HashSet::new(), 100), 2);
    assert_eq!(refill_rows(requests(), 4, 255, HashSet::new(), 100), 1);
    assert_eq!(refill_rows(requests(), 4, 0, HashSet::new(), 100), 0);
    assert_eq!(refill_rows(requests(), 0, 512, HashSet::new(), 100), 0);
}

#[test]
fn never_overtakes_an_older_long_request() {
    let requests = [128, 8192, 128].into_iter().map(|tokens| (tokens, &[][..]));
    assert_eq!(refill_rows(requests, 5, 512, HashSet::new(), 100), 1);
}

#[test]
fn counts_existing_prefill_pages_and_shared_prefix_pages() {
    use runtime::kv::BlockId;
    let resident = [BlockId(1), BlockId(2)].into_iter().collect::<HashSet<_>>();
    let shared = [BlockId(2), BlockId(3)];
    let extra = [BlockId(4)];
    assert_eq!(
        refill_rows(
            [(128, shared.as_slice()), (128, extra.as_slice())].into_iter(),
            4,
            512,
            resident,
            3
        ),
        1
    );
}
