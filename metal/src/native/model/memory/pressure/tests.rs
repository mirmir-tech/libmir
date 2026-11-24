use super::*;

#[test]
fn shrinks_scalar_prefill_only_under_allocator_pressure() {
    const GIB: usize = 1024 * 1024 * 1024;
    let memory = crate::engine::MemoryStats {
        active: 58 * GIB,
        cached: 2 * GIB,
        peak: 0,
        limit: 64 * GIB,
        recommended: Some(64 * GIB),
    };
    assert_eq!(pressure_bounded_budget(memory, 512, false), 128);
    assert_eq!(
        pressure_bounded_budget(
            crate::engine::MemoryStats { active: 40 * GIB, ..memory },
            512,
            false,
        ),
        512
    );
    assert_eq!(pressure_bounded_budget(memory, 512, true), 128);
}

#[test]
fn tunes_prefill_cohorts_to_live_memory_headroom() {
    const GIB: usize = 1024 * 1024 * 1024;
    let roomy = crate::engine::MemoryStats {
        active: 20 * GIB,
        cached: 0,
        peak: 0,
        limit: 64 * GIB,
        recommended: Some(56 * GIB),
    };
    let pressured = crate::engine::MemoryStats { active: 38 * GIB, ..roomy };
    let roomy_budget = prefill_token_budgets(roomy, 20 * 1024, 0);
    let pressured_budget = prefill_token_budgets(pressured, 20 * 1024, 0);
    let cached_budget = prefill_token_budgets(
        crate::engine::MemoryStats { cached: 8 * GIB, ..roomy },
        20 * 1024,
        0,
    );
    assert!(roomy_budget.0 > pressured_budget.0);
    assert!(roomy_budget.1 > pressured_budget.1);
    assert!(roomy_budget.1 >= roomy_budget.0 * PREFILL_WAVE_HEADROOM_DIVISOR);
    assert_eq!(cached_budget, roomy_budget);
    // Free arena pages extend both budgets even without fresh headroom.
    let full = crate::engine::MemoryStats { active: 56 * GIB, ..roomy };
    assert_eq!(prefill_token_budgets(full, 20 * 1024, 0), (1, 1));
    assert_eq!(prefill_token_budgets(full, 20 * 1024, 8192), (8192, 8192));
}

#[test]
fn keeps_history_clear_of_the_prefill_still_to_come() {
    const GIB: usize = 1024 * 1024 * 1024;
    let memory = crate::engine::MemoryStats {
        active: 36 * GIB,
        cached: 0,
        peak: 0,
        limit: 64 * GIB,
        recommended: Some(56 * GIB),
    };
    // 20 GiB free, 7 GiB reserve: 11 GiB of queued prompts leaves 2 GiB.
    assert!(!history_pressure(memory, 0));
    assert!(!history_pressure(memory, 11 * GIB));
    assert!(history_pressure(memory, 14 * GIB));
    assert_eq!(history_limit_bytes(memory, GIB, 0), 14 * GIB);
    assert_eq!(history_limit_bytes(memory, GIB, 11 * GIB), 3 * GIB);
    assert_eq!(history_limit_bytes(memory, GIB, 20 * GIB), 0);
}

#[test]
fn reclaims_unleased_prefixes_only_above_pressure_threshold() {
    let memory = crate::engine::MemoryStats {
        active: 50,
        cached: 6,
        peak: 0,
        limit: 100,
        recommended: Some(100),
    };
    assert!(prefix_reclamation_needed(memory));
    assert!(!prefix_reclamation_needed(crate::engine::MemoryStats {
        active: 44,
        cached: 6,
        ..memory
    }));
}
