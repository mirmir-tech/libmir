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
        if let Some(bytes) = other.measured_page_bytes()? {
            sized.push((other, bytes));
        }
    }
    if sized.is_empty() {
        return Ok(false);
    }
    let total = sized
        .iter()
        .fold(measured.page_bytes, |sum, (_, bytes)| sum.saturating_add(*bytes));
    let share = total / (u64::try_from(sized.len()).unwrap_or(u64::MAX).saturating_add(1));
    let mut shrank = false;
    for (other, bytes) in sized {
        if bytes <= share {
            continue;
        }
        let Some(target) = other.page_target(share) else {
            continue;
        };
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
    /// The measured budget of this model at `page_bytes` of pages, keeping
    /// the checkpoint budget equal to the pages.
    fn page_target(&self, page_bytes: u64) -> Option<MeasuredKvBudget> {
        let target = self.inner.engine.target();
        let estimate = self.inner.descriptor.memory_estimate_for(&self.inner.config, &target);
        let block_size = u64::try_from(self.inner.config.kv_cache.block_size).unwrap_or(u64::MAX);
        let block_bytes = estimate.kv_bytes_per_token.saturating_mul(block_size);
        if block_bytes == 0 {
            return None;
        }
        let blocks = (page_bytes / block_bytes).max(1).min(u64::from(u32::MAX));
        let page_bytes = block_bytes.saturating_mul(blocks);
        Some(MeasuredKvBudget {
            blocks: u32::try_from(blocks).unwrap_or(u32::MAX),
            page_bytes,
            checkpoint_bytes: page_bytes,
            budget_bytes: page_bytes.saturating_mul(2),
            available_bytes: 0,
        })
    }

    /// Grows an idle, measured model up to what memory now allows, for
    /// example after another model was unloaded.
    pub(in crate::model) fn regrow_kv_cache(&self) -> Result<bool> {
        let KvSizing::Measured(current) = self.kv_sizing()? else {
            return Ok(false);
        };
        let Some(measured) = self.measure_kv_budget(current.blocks)? else {
            return Ok(false);
        };
        // Reallocation drops retained prefixes; only a clear gain is worth it.
        if measured.blocks <= current.blocks.saturating_add(current.blocks / 8) {
            return Ok(false);
        }
        match self.resize_to(measured.blocks) {
            Ok(true) => {
                self.set_kv_sizing(KvSizing::Measured(measured))?;
                tracing::info!(
                    model = %self.inner.handle.id,
                    from_blocks = current.blocks,
                    to_blocks = measured.blocks,
                    "grew K/V cache into memory released by another model"
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
