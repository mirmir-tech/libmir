use std::cmp::max;

use super::{MemoryStats, Result, Stream, memory_stats};
use crate::FusionMode;

const MINIMUM_RESERVE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const RESERVE_DIVISOR: usize = 10;

pub trait ExpertFusion {
    fn enable_expert_fusion(&mut self, stream: &Stream) -> Result<bool>;
    fn expert_fusion_bytes(&self) -> Result<Option<usize>>;
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertFusionDecision {
    mode: FusionMode,
    enabled: bool,
    active: usize,
    additional: Option<usize>,
    recommended: Option<usize>,
    reserve: Option<usize>,
}

impl ExpertFusionDecision {
    pub fn summary(self) -> String {
        let additional =
            self.additional.map_or_else(|| "unavailable".into(), |bytes| bytes.to_string());
        let recommended =
            self.recommended.map_or_else(|| "unavailable".into(), |bytes| bytes.to_string());
        let reserve = self.reserve.map_or_else(|| "unavailable".into(), |bytes| bytes.to_string());
        let state = if self.enabled {
            "enabled"
        } else {
            "skipped"
        };
        let mode = match self.mode {
            FusionMode::Auto => "auto",
            FusionMode::Enabled => "forced",
            FusionMode::Disabled => "disabled",
        };
        format!(
            "expert gate/up fusion {state} ({mode}); active={} bytes, additional={additional} bytes, reserve={reserve} bytes, recommended={recommended} bytes",
            self.active
        )
    }
}

pub fn configure_expert_fusion<T: ExpertFusion>(
    layers: &mut [T],
    stream: &Stream,
    mode: FusionMode,
) -> Result<ExpertFusionDecision> {
    stream.synchronize()?;
    let memory = memory_stats()?;
    let additional = estimate_additional_bytes(layers)?;
    let reserve = memory.recommended.map(reserve_bytes);
    let enabled = match mode {
        FusionMode::Auto => additional.is_some_and(|bytes| fits_budget(memory, bytes)),
        FusionMode::Enabled => additional.is_some(),
        FusionMode::Disabled => false,
    };
    if enabled {
        for layer in layers {
            let _fused = layer.enable_expert_fusion(stream)?;
        }
        stream.synchronize()?;
    }
    Ok(ExpertFusionDecision {
        mode,
        enabled,
        active: memory.active,
        additional,
        recommended: memory.recommended,
        reserve,
    })
}

fn estimate_additional_bytes<T: ExpertFusion>(layers: &[T]) -> Result<Option<usize>> {
    let mut total = 0_usize;
    for layer in layers {
        let Some(bytes) = layer.expert_fusion_bytes()? else {
            return Ok(None);
        };
        total = total.checked_add(bytes).ok_or(super::Error::ShapeOverflow)?;
    }
    Ok(Some(total))
}

fn fits_budget(memory: MemoryStats, additional: usize) -> bool {
    let Some(recommended) = memory.recommended else {
        return false;
    };
    memory
        .active
        .checked_add(additional)
        .and_then(|used| used.checked_add(reserve_bytes(recommended)))
        .is_some_and(|required| required <= recommended)
}

fn reserve_bytes(recommended: usize) -> usize {
    max(MINIMUM_RESERVE_BYTES, recommended / RESERVE_DIVISOR)
}

#[cfg(test)]
mod tests {
    use super::{MemoryStats, fits_budget, reserve_bytes};

    #[test]
    fn auto_budget_keeps_memory_reserve() {
        let recommended = 32 * 1024 * 1024 * 1024;
        let memory = MemoryStats {
            active: 18 * 1024 * 1024 * 1024,
            cached: 0,
            peak: 0,
            limit: recommended,
            recommended: Some(recommended),
        };
        assert!(fits_budget(memory, 10 * 1024 * 1024 * 1024));
        assert!(!fits_budget(memory, 12 * 1024 * 1024 * 1024));
        assert_eq!(reserve_bytes(recommended), 3_435_973_836);
    }
}
