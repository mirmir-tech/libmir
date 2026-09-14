pub(super) fn reusable_prefill_budget(remaining: usize, capacity: usize) -> usize {
    const PLAN_ALIGNMENT: usize = 64;
    let count = remaining.min(capacity);
    // Decode rows consume part of the scheduler budget. Quantize large scratch
    // shapes to the GDN tile size instead of rebuilding every layer for 1023,
    // 1022, etc. Preserve small packed work and exact short checkpoint tails.
    if count > super::batch::PACKED_PREFILL_TOKEN_LIMIT {
        count / PLAN_ALIGNMENT * PLAN_ALIGNMENT
    } else {
        count
    }
}

#[cfg(test)]
mod tests {
    use super::reusable_prefill_budget;

    #[test]
    fn interleaved_rows_reuse_large_shapes_without_exceeding_the_budget() {
        for budget in 1008..1024 {
            assert_eq!(reusable_prefill_budget(budget, 2096), 960);
        }
        assert_eq!(reusable_prefill_budget(8192, 1024), 1024);
        assert_eq!(reusable_prefill_budget(8192, 2096), 2048);
        for tail in 0..=512 {
            assert_eq!(reusable_prefill_budget(tail, 2096), tail);
        }
        for budget in 1..=4096 {
            let count = reusable_prefill_budget(8192, budget);
            assert!(count > 0 && count <= budget);
        }
    }
}
