use super::{LoadedModel, clear_memory_cache, memory_stats, usable_memory};
use crate::native::error::Result;

const PRESSURE_PREFILL_TOKENS: usize = 128;
const KV_RATE_REFERENCE_TOKENS: usize = 16 * 1_024;
const PREFILL_MEMORY_RESERVE_DIVISOR: usize = 8;
const PREFILL_MINIMUM_RESERVE: usize = 2 * 1024 * 1024 * 1024;
/// Share of the headroom a physical wave may hold as whole-prompt K/V; the
/// packed workspace check still guards each chunk's activations.
const PREFILL_WAVE_HEADROOM_DIVISOR: usize = 2;
const PREFIX_RECLAIM_PRESSURE_PERCENT: usize = 50;

impl LoadedModel {
    pub(crate) fn pressure_bounded_prefill_budget(
        budget: usize,
        workspace_constrained: bool,
    ) -> Result<usize> {
        Ok(pressure_bounded_budget(memory_stats()?, budget, workspace_constrained))
    }

    /// K/V bytes a long prompt adds per token, sliding windows charged once.
    pub(crate) fn kv_rate_bytes_per_token(&self) -> Result<usize> {
        Ok(self
            .estimated_prefix_bytes(KV_RATE_REFERENCE_TOKENS)?
            .div_ceil(KV_RATE_REFERENCE_TOKENS)
            .max(1))
    }

    pub(crate) fn prefill_memory_token_budgets(&self) -> Result<(usize, usize)> {
        let memory = memory_stats()?;
        // One token makes every sliding window look unbounded; a long-prompt
        // average charges windows once (Gemma: 225 KB/token at 1, 33 KB at 16k).
        let kv_bytes_per_token = self.kv_rate_bytes_per_token()?;
        let budgets = prefill_token_budgets(memory, kv_bytes_per_token);
        tracing::debug!(
            active_bytes = memory.active,
            cached_bytes = memory.cached,
            usable_bytes = usable_memory(memory),
            kv_bytes_per_token,
            max_wave_tokens = budgets.0,
            max_cohort_tokens = budgets.1,
            "updated memory-bounded Metal prefill limits"
        );
        Ok(budgets)
    }

    pub(crate) fn reclaim_unleased_prefixes_for_prefill(&mut self) -> Result<bool> {
        let before = memory_stats()?;
        if self.prefixes.group_count() == 0 || !prefix_reclamation_needed(before) {
            return Ok(false);
        }
        let groups = self.prefixes.group_count();
        let bytes = self.prefixes.resident_bytes();
        self.prefixes.clear();
        clear_memory_cache()?;
        let after = memory_stats()?;
        tracing::debug!(
            groups,
            prefix_bytes = bytes,
            active_bytes_before = before.active,
            cached_bytes_before = before.cached,
            active_bytes_after = after.active,
            cached_bytes_after = after.cached,
            "reclaimed unleased Metal prefixes before prefill"
        );
        Ok(true)
    }
}

pub(super) fn prefix_reclamation_needed(memory: crate::engine::MemoryStats) -> bool {
    let usable = usable_memory(memory);
    usable > 0
        && memory.active.saturating_add(memory.cached)
            > usable.saturating_mul(PREFIX_RECLAIM_PRESSURE_PERCENT) / 100
}

/// Live allocations only: the allocator cache is reclaimable. Paged arenas keep
/// their high-water capacity, so a share of `active` says little; suspend only
/// when the real headroom no longer covers the prefill reserve.
pub(super) fn history_pressure(memory: crate::engine::MemoryStats, pending: usize) -> bool {
    let usable = usable_memory(memory);
    usable > 0
        && usable.saturating_sub(memory.active) < prefill_reserve(usable).saturating_add(pending)
}

/// The whole headroom above the prefill reserve; the history already retained
/// is part of `active` and counts as available to itself.
pub(super) fn history_limit_bytes(
    memory: crate::engine::MemoryStats,
    retained: usize,
    pending: usize,
) -> usize {
    let usable = usable_memory(memory);
    let live = memory.active.saturating_sub(retained);
    usable
        .saturating_sub(live)
        .saturating_sub(prefill_reserve(usable))
        .saturating_sub(pending)
}

fn prefill_reserve(usable: usize) -> usize {
    (usable / PREFILL_MEMORY_RESERVE_DIVISOR).max(PREFILL_MINIMUM_RESERVE)
}

fn prefill_token_budgets(
    memory: crate::engine::MemoryStats,
    kv_bytes_per_token: usize,
) -> (usize, usize) {
    let usable = usable_memory(memory);
    if usable == 0 {
        return (1, 1);
    }
    let reserve = prefill_reserve(usable);
    let headroom = usable.saturating_sub(memory.active).saturating_sub(reserve);
    let kv_bytes = kv_bytes_per_token.max(1);
    // Workspace scales with a chunk, not with the whole prompt: sizing waves by
    // sixteen prompt-length copies admitted one 32k row at a time.
    let cohort_tokens = headroom.checked_div(kv_bytes).unwrap_or(0).max(1);
    let wave_tokens = (cohort_tokens / PREFILL_WAVE_HEADROOM_DIVISOR).max(1);
    (wave_tokens, cohort_tokens)
}

fn pressure_bounded_budget(
    memory: crate::engine::MemoryStats,
    budget: usize,
    workspace_constrained: bool,
) -> usize {
    let usable = usable_memory(memory);
    // The allocator cache is reclaimable, and paged arenas keep their high-water
    // capacity, so only headroom below the prefill reserve shrinks a chunk.
    let available = usable.saturating_sub(memory.active);
    if workspace_constrained || usable > 0 && available < prefill_reserve(usable) {
        budget.min(PRESSURE_PREFILL_TOKENS)
    } else {
        budget
    }
}

#[cfg(test)]
mod tests {
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
        let roomy_budget = prefill_token_budgets(roomy, 20 * 1024);
        let pressured_budget = prefill_token_budgets(pressured, 20 * 1024);
        let cached_budget = prefill_token_budgets(
            crate::engine::MemoryStats { cached: 8 * GIB, ..roomy },
            20 * 1024,
        );
        assert!(roomy_budget.0 > pressured_budget.0);
        assert!(roomy_budget.1 > pressured_budget.1);
        assert!(roomy_budget.1 >= roomy_budget.0 * PREFILL_WAVE_HEADROOM_DIVISOR);
        assert_eq!(cached_budget, roomy_budget);
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
}
