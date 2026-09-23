//! Moving K/V pages between loaded models.

use super::{KvSizing, Model, automatic_cache::MeasuredKvBudget, registry::MINIMUM_SHARE_TOKENS};
use crate::{Result, RuntimeConfig};

/// Whether `measured` leaves the model unable to serve one long request.
pub(super) fn below_minimum_share(config: &RuntimeConfig, measured: MeasuredKvBudget) -> bool {
    let block_size = u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX);
    u64::from(measured.blocks).saturating_mul(block_size) < MINIMUM_SHARE_TOKENS
}

/// Shrinks idle models loaded before `model` to an equal share of the pages
/// they and `model` would hold together. Returns whether any model shrank.
pub(super) fn shrink_others(model: &Model, measured: MeasuredKvBudget) -> Result<bool> {
    let others = model.inner.registry.others(&model.inner)?;
    let mut sized = Vec::with_capacity(others.len());
    for other in others {
        if let Some(budget) = other.measured_budget()? {
            sized.push((other, budget));
        }
    }
    if sized.is_empty() {
        return Ok(false);
    }
    let total = sized
        .iter()
        .fold(measured.page_bytes, |sum, (_, budget)| sum.saturating_add(budget.page_bytes));
    let share = total / (u64::try_from(sized.len()).unwrap_or(u64::MAX).saturating_add(1));
    let mut shrank = false;
    for (other, budget) in sized {
        let bytes = budget.page_bytes;
        if bytes <= share {
            continue;
        }
        let Some(mut target) = Model::page_target(share, budget) else {
            continue;
        };
        target.others_bytes = other.inner.memory.others_bytes()?;
        match other.resize_to(target.blocks) {
            Ok(true) => {
                other.set_kv_sizing(KvSizing::Measured(target))?;
                tracing::info!(
                    model = %other.inner.handle.id,
                    from_bytes = bytes,
                    to_bytes = target.page_bytes,
                    "shrank K/V cache to share pages with a newly loaded model"
                );
                shrank = true;
            },
            Ok(false) => {},
            Err(error) => tracing::info!(
                model = %other.inner.handle.id,
                %error,
                "left a busy model's K/V cache alone while sharing pages"
            ),
        }
    }
    Ok(shrank)
}

impl Model {
    /// This model's budget at `page_bytes` of pages, within the block cap
    /// of `current`, with the checkpoint share that follows the pages.
    fn page_target(page_bytes: u64, current: MeasuredKvBudget) -> Option<MeasuredKvBudget> {
        let block_bytes = current.page_bytes / u64::from(current.blocks.max(1));
        if block_bytes == 0 {
            return None;
        }
        let blocks = (page_bytes / block_bytes).max(1).min(u64::from(current.max_blocks.max(1)));
        let page_bytes = block_bytes.saturating_mul(blocks);
        let checkpoint_bytes = super::checkpoint_bytes(page_bytes);
        Some(MeasuredKvBudget {
            blocks: u32::try_from(blocks).unwrap_or(u32::MAX),
            page_bytes,
            checkpoint_bytes,
            budget_bytes: page_bytes.saturating_add(checkpoint_bytes),
            ..current
        })
    }

    /// Bytes this model's measured cache could give back to another load:
    /// its pages and checkpoints above the minimum share.
    pub(in crate::model) fn reclaimable_kv_bytes(&self) -> Result<u64> {
        let KvSizing::Measured(current) = self.kv_sizing()? else {
            return Ok(0);
        };
        let held = current.page_bytes.saturating_add(current.checkpoint_bytes);
        let block_size = u64::try_from(self.inner.config.kv_cache.block_size).unwrap_or(u64::MAX);
        let block_bytes = current.page_bytes / u64::from(current.blocks.max(1));
        let minimum_pages =
            MINIMUM_SHARE_TOKENS.div_ceil(block_size.max(1)).saturating_mul(block_bytes);
        let minimum = minimum_pages.saturating_add(super::checkpoint_bytes(minimum_pages));
        Ok(held.saturating_sub(minimum))
    }

    /// Resizes an idle, measured model as other models' memory changes: its
    /// pages follow their reservations one for one (keeping the checkpoint
    /// share), so it shrinks after a load and grows back after an unload.
    /// The measurement taken at sizing stays the anchor; measuring again
    /// under traffic would follow retained shapes and checkpoints instead.
    pub(in crate::model) fn rebalance_kv_cache(&self) -> Result<bool> {
        let KvSizing::Measured(current) = self.kv_sizing()? else {
            return Ok(false);
        };
        let others = self.inner.memory.others_bytes()?;
        if others == current.others_bytes {
            return Ok(false);
        }
        let divisor =
            u64::try_from(runtime::kv::PREFIX_CHECKPOINT_PAGE_DIVISOR).unwrap_or(u64::MAX);
        let follow = |delta: u64| delta / divisor.saturating_add(1) * divisor;
        let pages = if others > current.others_bytes {
            current.page_bytes.saturating_sub(follow(others - current.others_bytes))
        } else {
            current.page_bytes.saturating_add(follow(current.others_bytes - others))
        };
        let Some(mut measured) = Self::page_target(pages, current) else {
            return Ok(false);
        };
        measured.others_bytes = others;
        // Reallocation drops retained prefixes; only a clear change is worth
        // it. A smaller change is remembered against the next one.
        let clear = measured.blocks > current.blocks.saturating_add(current.blocks / 8)
            || measured.blocks < current.blocks.saturating_sub(current.blocks / 8);
        if !clear {
            return Ok(false);
        }
        match self.resize_to(measured.blocks) {
            Ok(true) => {
                self.set_kv_sizing(KvSizing::Measured(measured))?;
                tracing::info!(
                    model = %self.inner.handle.id,
                    from_blocks = current.blocks,
                    to_blocks = measured.blocks,
                    others_bytes = others,
                    "resized K/V cache to follow other models' memory"
                );
                Ok(true)
            },
            Ok(false) => Ok(false),
            Err(error) => {
                tracing::info!(model = %self.inner.handle.id, %error, "left a busy model's K/V cache alone");
                Ok(false)
            },
        }
    }
}
