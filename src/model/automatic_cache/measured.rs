//! K/V sizing from memory measured after warm-up, for backends that can
//! reallocate pages of an idle model.
//!
//! An estimate cannot see what a runtime keeps beside the weights: packed
//! projection copies, execution plans, tuning scratch and allocator
//! retention. Loading with a provisional cache, warming the execution
//! profiles, and then measuring what is left makes those costs part of the
//! measurement instead of a guess.

use super::{ADMISSION_SNAPSHOT_HEADROOM_BYTES, memory_policy, unified_limit, useful_blocks};
use crate::{MemorySnapshot, ModelMemoryEstimate, RuntimeConfig};

/// Tokens the warm-up workload holds at once: its two profile sessions and
/// the exact prefill shapes, with a decode block each.
const PROVISIONAL_SESSION_TOKENS: u64 = 2 * (2_048 + 64);
const PROVISIONAL_SHAPE_SESSIONS: u64 = 4;
/// Backends warm exact prefill shapes up to their default chunk.
const PROVISIONAL_CHUNK_TOKENS: u64 = 2_096;
/// Concurrency warm-up prefills two chunks into each of `max_batch_requests`
/// sessions and decodes one token.
const PROVISIONAL_CONCURRENCY_TOKENS: u64 = 2 * PROVISIONAL_CHUNK_TOKENS + 64;

/// Costs of traffic that the warm-up leaves resident only in part.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrafficSlack {
    /// What the backend's retained execution shapes may still grow by.
    pub(crate) retention_headroom_bytes: u64,
    /// What one live session costs beside its K/V pages.
    pub(crate) session_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeasuredKvBudget {
    pub(crate) blocks: u32,
    /// Bytes of the K/V pages at `blocks`.
    pub(crate) page_bytes: u64,
    /// Bytes the backend may retain as prefix checkpoints beside the pages.
    pub(crate) checkpoint_bytes: u64,
    /// Memory left after the platform reserve and traffic slack.
    pub(crate) budget_bytes: u64,
    /// Host memory available at measurement, with the provisional pages added.
    pub(crate) available_bytes: u64,
}

/// Blocks for loading and warming a model whose final cache is measured
/// afterwards: enough for the warm-up workload, never more than the
/// estimate-based automatic size.
pub(in crate::model) fn provisional_blocks(config: &RuntimeConfig, estimated: u32) -> u32 {
    let block_size = u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX).max(1);
    let shape_tokens = u64::try_from(config.scheduler.max_batch_tokens)
        .unwrap_or(u64::MAX)
        .max(PROVISIONAL_CHUNK_TOKENS)
        .saturating_mul(PROVISIONAL_SHAPE_SESSIONS);
    let concurrency_tokens = u64::try_from(config.scheduler.max_batch_requests)
        .unwrap_or(u64::MAX)
        .saturating_mul(PROVISIONAL_CONCURRENCY_TOKENS);
    let tokens = PROVISIONAL_SESSION_TOKENS
        .saturating_add(shape_tokens)
        .saturating_add(concurrency_tokens);
    let blocks = tokens.div_ceil(block_size).max(1).min(u64::from(u32::MAX));
    u32::try_from(blocks).unwrap_or(u32::MAX).min(estimated).max(1)
}

/// Fraction of unified memory kept free by measured sizing: everything the
/// runtime holds is already counted, so only the host and drift remain.
const MEASURED_RESERVE_DIVISOR: u64 = 8;
const MEASURED_RESERVE_FLOOR_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// The platform reserve of measured sizing. An explicit operator reserve
/// stands; the default keeps an eighth of the memory (at least 8 GiB) plus
/// the allocator's own reserve, not the quarter the estimate path needs to
/// cover what it cannot see.
fn measured_reserve(config: &RuntimeConfig, memory: &MemorySnapshot) -> u64 {
    if config.memory.reserve_percent.is_some() || config.memory.reserve_bytes.is_some() {
        return memory_policy::platform_reserve(config.memory, memory);
    }
    memory
        .total_bytes
        .map_or(MEASURED_RESERVE_FLOOR_BYTES, |total| {
            (total / MEASURED_RESERVE_DIVISOR).max(MEASURED_RESERVE_FLOOR_BYTES)
        })
        .saturating_add(memory.allocation_reserve_bytes)
}

/// Sizes the cache from `memory` measured while `provisional_blocks` pages
/// are still allocated. `slack` is what traffic still adds: retained shapes
/// growing and one session's cost for every admissible request. Pages and
/// prefix checkpoints split the budget evenly, because the backend bounds
/// checkpoints by the page bytes.
pub(in crate::model) fn measured(
    config: &RuntimeConfig,
    estimate: ModelMemoryEstimate,
    memory: &MemorySnapshot,
    provisional_blocks: u32,
    slack: TrafficSlack,
) -> Option<MeasuredKvBudget> {
    let block_size = u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX).max(1);
    let block_bytes = estimate.kv_bytes_per_token.saturating_mul(block_size);
    if block_bytes == 0 {
        return None;
    }
    let provisional_bytes = block_bytes.saturating_mul(u64::from(provisional_blocks));
    // Memory cached by the backend allocator is not counted as available:
    // freed shapes rarely match the next allocation, so it stays reserved.
    let available = memory.available_bytes?.saturating_add(provisional_bytes);
    let reserve = measured_reserve(config, memory);
    let sessions = u64::try_from(config.scheduler.max_batch_requests).unwrap_or(u64::MAX);
    let slack = slack
        .session_bytes
        .saturating_mul(sessions)
        .saturating_add(slack.retention_headroom_bytes)
        .saturating_add(ADMISSION_SNAPSHOT_HEADROOM_BYTES);
    let mut budget = available.saturating_sub(reserve).saturating_sub(slack);
    if let Some(limit) = unified_limit(memory) {
        budget = budget.min(limit);
    }
    let pages = budget / 2;
    let blocks = (pages / block_bytes)
        .min(useful_blocks(config, estimate))
        .max(1)
        .min(u64::from(u32::MAX));
    let page_bytes = block_bytes.saturating_mul(blocks);
    Some(MeasuredKvBudget {
        blocks: u32::try_from(blocks).unwrap_or(u32::MAX),
        page_bytes,
        checkpoint_bytes: page_bytes,
        budget_bytes: budget,
        available_bytes: available,
    })
}
