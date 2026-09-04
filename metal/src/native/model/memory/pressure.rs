use super::{LoadedModel, clear_memory_cache, memory_stats, usable_memory};
use crate::native::error::Result;

const PRESSURE_DIVISOR: usize = 2;
const PRESSURE_PREFILL_TOKENS: usize = 128;
const PREFILL_MEMORY_RESERVE_DIVISOR: usize = 8;
const PREFILL_MINIMUM_RESERVE: usize = 2 * 1024 * 1024 * 1024;
const PREFILL_WORKSPACE_KV_COPIES: usize = 16;
const PREFIX_RECLAIM_PRESSURE_PERCENT: usize = 50;

impl LoadedModel {
    pub(crate) fn pressure_bounded_prefill_budget(
        budget: usize,
        workspace_constrained: bool,
    ) -> Result<usize> {
        Ok(pressure_bounded_budget(memory_stats()?, budget, workspace_constrained))
    }

    pub(crate) fn prefill_memory_token_budgets(&self) -> Result<(usize, usize)> {
        let memory = memory_stats()?;
        let kv_bytes_per_token = self.estimated_prefix_bytes(1)?.max(1);
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

fn prefix_reclamation_needed(memory: crate::engine::MemoryStats) -> bool {
    let usable = usable_memory(memory);
    usable > 0
        && memory.active.saturating_add(memory.cached)
            > usable.saturating_mul(PREFIX_RECLAIM_PRESSURE_PERCENT) / 100
}

fn prefill_token_budgets(
    memory: crate::engine::MemoryStats,
    kv_bytes_per_token: usize,
) -> (usize, usize) {
    let usable = usable_memory(memory);
    if usable == 0 {
        return (1, 1);
    }
    let reserve = (usable / PREFILL_MEMORY_RESERVE_DIVISOR).max(PREFILL_MINIMUM_RESERVE);
    let headroom = usable.saturating_sub(memory.active).saturating_sub(reserve);
    let kv_bytes = kv_bytes_per_token.max(1);
    let wave_bytes = kv_bytes.saturating_mul(PREFILL_WORKSPACE_KV_COPIES).max(1);
    let wave_tokens = headroom.checked_div(wave_bytes).unwrap_or(0).max(1);
    (wave_tokens, wave_tokens)
}

fn pressure_bounded_budget(
    memory: crate::engine::MemoryStats,
    budget: usize,
    workspace_constrained: bool,
) -> usize {
    let usable = usable_memory(memory);
    let retained = memory.active.saturating_add(memory.cached);
    let available = usable.saturating_sub(retained);
    if workspace_constrained || usable > 0 && available < usable / PRESSURE_DIVISOR {
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
        let memory = crate::engine::MemoryStats {
            active: 40,
            cached: 12,
            peak: 0,
            limit: 64,
            recommended: Some(64),
        };
        assert_eq!(pressure_bounded_budget(memory, 512, false), 128);
        assert_eq!(
            pressure_bounded_budget(
                crate::engine::MemoryStats { active: 16, ..memory },
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
        assert_eq!(roomy_budget.1, roomy_budget.0);
        assert_eq!(cached_budget, roomy_budget);
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
