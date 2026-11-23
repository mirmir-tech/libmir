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
        // Arenas keep their high-water capacity inside `active`: their free
        // pages take new K/V without allocating.
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        let reusable_tokens = self.allocated_page_headroom()?.saturating_mul(page_size);
        let budgets = prefill_token_budgets(memory, kv_bytes_per_token, reusable_tokens);
        tracing::debug!(
            active_bytes = memory.active,
            cached_bytes = memory.cached,
            usable_bytes = usable_memory(memory),
            kv_bytes_per_token,
            reusable_tokens,
            max_wave_tokens = budgets.0,
            max_cohort_tokens = budgets.1,
            "updated memory-bounded Metal prefill limits"
        );
        Ok(budgets)
    }

    /// Clears unleased prefixes only when the cohort's K/V (`pending_tokens`)
    /// fits neither the headroom nor the arenas' free pages: their pages stay
    /// in arena capacity, so clearing for a high weight share alone freed
    /// nothing (2 GB of Qwen3-30B `Q8_0` snapshots, 1.8 MB active) and dropped
    /// contexts that queued requests were about to restore.
    pub(crate) fn reclaim_unleased_prefixes_for_prefill(
        &mut self,
        pending_tokens: usize,
    ) -> Result<bool> {
        let before = memory_stats()?;
        if self.prefixes.group_count() == 0 || !prefix_reclamation_needed(before) {
            return Ok(false);
        }
        let page_size = self.stream.config().kv_cache.block_size.max(1);
        let reusable = self.allocated_page_headroom()?.saturating_mul(page_size);
        let fresh = prefill_token_budgets(before, self.kv_rate_bytes_per_token()?, reusable).1;
        if fresh >= pending_tokens {
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
    reusable_tokens: usize,
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
    let fresh_tokens = headroom.checked_div(kv_bytes).unwrap_or(0);
    // Free arena pages are already allocated (they count in `active`), so they
    // take whole-prompt K/V without the wave's headroom share: Qwen3-30B Q8_0
    // at depth 4096 C10 otherwise prefilled four of ten rows one at a time.
    let cohort_tokens = fresh_tokens.saturating_add(reusable_tokens).max(1);
    let wave_tokens = (fresh_tokens / PREFILL_WAVE_HEADROOM_DIVISOR)
        .saturating_add(reusable_tokens)
        .max(1);
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
mod tests;
