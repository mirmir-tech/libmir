use std::cell::Cell;

thread_local! {
    static COUNTS: Cell<[usize; 2]> = const { Cell::new([0, 0]) };
}

pub fn counts() -> [usize; 2] {
    COUNTS.get()
}

pub(super) fn record(append: bool) {
    COUNTS.set({
        let mut counts = COUNTS.get();
        counts[usize::from(append)] += 1;
        counts
    });
}
