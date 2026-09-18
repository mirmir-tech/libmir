use std::collections::HashSet;

use runtime::kv::BlockId;

pub(super) fn refill_rows<'a>(
    requests: impl Iterator<Item = (usize, &'a [BlockId])>,
    slots: usize,
    token_budget: usize,
    mut resident: HashSet<BlockId>,
    capacity: usize,
) -> usize {
    let mut tokens = 0_usize;
    let mut count = 0;
    for (length, blocks) in requests.take(slots) {
        // Do not jump over an older long request or discount a long cache hit.
        if !(1..=128).contains(&length) || length > token_budget.saturating_sub(tokens) {
            break;
        }
        resident.extend(blocks.iter().copied());
        if resident.len() > capacity {
            break;
        }
        tokens += length;
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests;
