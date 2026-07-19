use models::layout::{PooledVisionConfig, SpatialMergeVisionConfig};

use super::Model;
use crate::{Error, Result, VisionRuntimeConfig};

const SCORE_ELEMENT_BYTES: u64 = 4;
const FALLBACK_ATTENTION_BUDGET: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub(super) struct VisionLimits {
    pub max_pixels: Option<usize>,
    pub attention_budget_bytes: u64,
}

impl Model {
    pub(super) fn vision_limits(&self) -> VisionLimits {
        let config = self.inner.config.vision;
        let available = self
            .inner
            .engine
            .memory_snapshot()
            .ok()
            .and_then(|memory| memory.available_bytes);
        VisionLimits {
            max_pixels: config.max_pixels,
            attention_budget_bytes: attention_budget(config, available),
        }
    }
}

impl VisionLimits {
    pub fn pooled_patch_limit(self, vision: &PooledVisionConfig) -> usize {
        let attention =
            attention_patch_limit(self.attention_budget_bytes, vision.num_attention_heads);
        let patch_area = vision.patch_size.saturating_mul(vision.patch_size).max(1);
        self.max_pixels.map_or(attention, |pixels| attention.min(pixels / patch_area))
    }

    pub fn spatial_pixel_limit(self, vision: &SpatialMergeVisionConfig) -> usize {
        let patches =
            attention_patch_limit(self.attention_budget_bytes, vision.num_attention_heads);
        let patch_area = vision.patch_size.saturating_mul(vision.patch_size);
        let attention_pixels = patches.saturating_mul(patch_area);
        self.max_pixels.map_or(attention_pixels, |pixels| pixels.min(attention_pixels))
    }

    pub fn validate(self, patch_tokens: usize, attention_heads: usize) -> Result<()> {
        let required = attention_bytes(patch_tokens, attention_heads);
        if required <= u128::from(self.attention_budget_bytes) {
            return Ok(());
        }
        Err(Error::VisionResourceLimit {
            patch_tokens,
            required_bytes: u64::try_from(required).unwrap_or(u64::MAX),
            budget_bytes: self.attention_budget_bytes,
        })
    }
}

fn attention_budget(config: VisionRuntimeConfig, available: Option<u64>) -> u64 {
    config.attention_budget_bytes.unwrap_or_else(|| {
        let percent = u64::from(config.memory_percent.clamp(1, 100));
        available
            .map(|bytes| bytes.saturating_mul(percent) / 100)
            .filter(|bytes| *bytes > 0)
            .unwrap_or(FALLBACK_ATTENTION_BUDGET)
    })
}

fn attention_patch_limit(budget: u64, heads: usize) -> usize {
    let divisor = SCORE_ELEMENT_BYTES.saturating_mul(u64::try_from(heads).unwrap_or(u64::MAX));
    let scores = budget / divisor.max(1);
    usize::try_from(integer_sqrt(scores)).unwrap_or(usize::MAX)
}

fn attention_bytes(patches: usize, heads: usize) -> u128 {
    let patches = patches as u128;
    u128::from(SCORE_ELEMENT_BYTES) * heads as u128 * patches * patches
}

fn integer_sqrt(value: u64) -> u64 {
    if value < 2 {
        return value;
    }
    let mut current = value;
    let mut next = u64::midpoint(current, value / current);
    while next < current {
        current = next;
        next = u64::midpoint(current, value / current);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_quadratic_patch_limit() {
        assert_eq!(attention_patch_limit(4 * 16 * 100 * 100, 16), 100);
        assert_eq!(attention_patch_limit(4 * 16 * 101 * 101 - 1, 16), 100);
    }

    #[test]
    fn fixed_budget_overrides_available_memory() {
        let config = VisionRuntimeConfig {
            attention_budget_bytes: Some(123),
            ..VisionRuntimeConfig::default()
        };
        assert_eq!(attention_budget(config, Some(10_000)), 123);
    }

    #[test]
    fn automatic_budget_uses_available_memory() {
        assert_eq!(attention_budget(VisionRuntimeConfig::default(), Some(10_000)), 8_000);
    }
}
