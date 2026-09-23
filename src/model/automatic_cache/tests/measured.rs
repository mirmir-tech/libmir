use super::{GIB, estimate, memory};
use crate::{
    RuntimeConfig,
    model::automatic_cache::{TrafficSlack, measured, provisional_blocks},
};

const MIB: u64 = 1024 * 1024;

/// Qwen3.8-27B on a 121 GiB GB10: 64 KiB per token, 151 MiB of recurrent
/// state per session, 39 GiB left after load and warm-up with 512 blocks.
#[test]
fn measured_budget_splits_pages_and_checkpoints_after_the_reserve() -> crate::Result<()> {
    let config = RuntimeConfig::default();
    let mut estimate = estimate(51 * GIB, 5 * GIB, 65_536, 262_144);
    estimate.session_state_bytes = 151 * MIB;
    let memory = memory(121 * GIB, 39 * GIB, 4 * GIB, true);

    let slack = TrafficSlack {
        session_bytes: estimate.session_state_bytes,
        ..Default::default()
    };
    let budget =
        measured(&config, estimate, &memory, 512, slack).ok_or(crate::Error::EmptyPrompt)?;
    let sessions = u64::try_from(config.scheduler.max_batch_requests).unwrap_or(u64::MAX);
    let reserve = 121 * GIB / 8 + 4 * GIB;
    let slack = estimate.session_state_bytes * sessions + 64 * MIB;
    let available = 39 * GIB + 512 * 16 * 65_536;

    assert_eq!(budget.budget_bytes, available - reserve - slack);
    // Pages take four fifths of the budget; checkpoints a quarter of the pages.
    assert_eq!(budget.checkpoint_bytes, budget.page_bytes / 4);
    assert!(budget.page_bytes <= budget.budget_bytes / 5 * 4);
    assert!(budget.page_bytes + 16 * 65_536 > budget.budget_bytes / 5 * 4);
    assert_eq!(u64::from(budget.blocks) * 16 * 65_536, budget.page_bytes);
    Ok(())
}

#[test]
fn measured_budget_is_capped_by_useful_tokens_and_unified_share() -> crate::Result<()> {
    let config = RuntimeConfig::default();
    let estimate = estimate(GIB, GIB, 1_024, 4_096);
    let memory = memory(128 * GIB, 120 * GIB, 0, true);

    let budget = measured(&config, estimate, &memory, 1, TrafficSlack::default())
        .ok_or(crate::Error::EmptyPrompt)?;
    let useful = 4_096 * u64::try_from(config.scheduler.max_batch_requests).unwrap_or(u64::MAX);

    assert_eq!(u64::from(budget.blocks) * 16, useful.div_ceil(16) * 16);
    assert!(budget.budget_bytes <= 128 * GIB / 5 * 2);
    Ok(())
}

#[test]
fn measured_budget_never_drops_below_one_block() {
    let config = RuntimeConfig::default();
    let estimate = estimate(60 * GIB, 6 * GIB, 65_536, 262_144);
    let memory = memory(64 * GIB, 8 * GIB, 0, true);

    assert_eq!(
        measured(&config, estimate, &memory, 4, TrafficSlack::default())
            .map(|budget| budget.blocks),
        Some(1)
    );
}

#[test]
fn provisional_blocks_cover_warm_up_within_the_estimate() {
    let config = RuntimeConfig::default();
    let shapes = u64::try_from(config.scheduler.max_batch_tokens.max(2_096)).unwrap_or(0);
    let rows = u64::try_from(config.scheduler.max_batch_requests).unwrap_or(0);
    let tokens = 2 * (2_048 + 64) + 4 * shapes + rows * (2 * 2_096 + 64);
    let expected = u32::try_from(tokens.div_ceil(16)).unwrap_or(u32::MAX);

    assert_eq!(provisional_blocks(&config, u32::MAX), expected);
    assert_eq!(provisional_blocks(&config, 100), 100);
    assert_eq!(provisional_blocks(&config, 0), 1);
}

#[test]
fn retention_headroom_comes_out_of_the_budget() -> crate::Result<()> {
    let config = RuntimeConfig::default();
    let estimate = estimate(51 * GIB, 5 * GIB, 65_536, 262_144);
    let memory = memory(121 * GIB, 39 * GIB, 4 * GIB, true);

    let plain = measured(&config, estimate, &memory, 512, TrafficSlack::default())
        .ok_or(crate::Error::EmptyPrompt)?;
    let retention = TrafficSlack {
        retention_headroom_bytes: 2 * GIB,
        session_bytes: 0,
    };
    let held =
        measured(&config, estimate, &memory, 512, retention).ok_or(crate::Error::EmptyPrompt)?;
    let sessions = TrafficSlack {
        retention_headroom_bytes: 0,
        session_bytes: 512 * MIB,
    };
    let occupied =
        measured(&config, estimate, &memory, 512, sessions).ok_or(crate::Error::EmptyPrompt)?;
    let rows = u64::try_from(config.scheduler.max_batch_requests).unwrap_or(u64::MAX);

    assert_eq!(plain.budget_bytes - held.budget_bytes, 2 * GIB);
    assert_eq!(plain.budget_bytes - occupied.budget_bytes, rows * 512 * MIB);
    assert!(held.blocks < plain.blocks);
    Ok(())
}
