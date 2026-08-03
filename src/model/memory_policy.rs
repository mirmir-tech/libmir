use crate::{MemoryRuntimeConfig, MemorySnapshot, ModelMemoryEstimate};

pub(super) fn platform_reserve(config: MemoryRuntimeConfig, memory: &MemorySnapshot) -> u64 {
    config.hard_reserve_bytes(memory)
}

pub(super) const fn transient_reserve(
    _estimate: ModelMemoryEstimate,
    _memory: &MemorySnapshot,
) -> u64 {
    // Runtime workspace is already included in `required_bytes`; reserving an
    // additional fraction of the weights would strand accelerator memory and
    // override the operator's explicit hard-reserve policy.
    0
}

pub(super) fn planned_residency(estimate: ModelMemoryEstimate, memory: &MemorySnapshot) -> u64 {
    estimate.required_bytes.saturating_add(transient_reserve(estimate, memory))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn unified_policy_does_not_duplicate_workspace_headroom() {
        let mut memory = snapshot(128 * GIB, true);
        memory.allocation_reserve_bytes = 4 * GIB;
        let estimate = estimate(40 * GIB);

        assert_eq!(platform_reserve(MemoryRuntimeConfig::default(), &memory), 36 * GIB);
        assert_eq!(transient_reserve(estimate, &memory), 0);
        assert_eq!(planned_residency(estimate, &memory), 40 * GIB);
    }

    #[test]
    fn discrete_device_uses_only_the_global_reserve() {
        let memory = snapshot(80 * GIB, false);
        let estimate = estimate(40 * GIB);

        assert_eq!(platform_reserve(MemoryRuntimeConfig::default(), &memory), 8 * GIB);
        assert_eq!(transient_reserve(estimate, &memory), 0);
        assert_eq!(planned_residency(estimate, &memory), 40 * GIB);
    }

    #[test]
    fn explicit_percentage_replaces_backend_default() {
        let mut memory = snapshot(128 * GIB, true);
        memory.allocation_reserve_bytes = 4 * GIB;
        let config = MemoryRuntimeConfig {
            reserve_percent: Some(1),
            reserve_bytes: None,
        };

        assert_eq!(platform_reserve(config, &memory), 128 * GIB / 100);
        assert_eq!(
            platform_reserve(
                MemoryRuntimeConfig {
                    reserve_percent: Some(0),
                    reserve_bytes: None,
                },
                &memory,
            ),
            0
        );
    }

    fn snapshot(total: u64, unified: bool) -> MemorySnapshot {
        MemorySnapshot {
            total_bytes: Some(total),
            available_bytes: Some(total),
            active_bytes: 0,
            cached_bytes: 0,
            allocation_reserve_bytes: 0,
            source: "test".into(),
            unified,
        }
    }

    const fn estimate(required_bytes: u64) -> ModelMemoryEstimate {
        ModelMemoryEstimate {
            weight_bytes: required_bytes,
            kv_cache_bytes: 0,
            workspace_bytes: 0,
            required_bytes,
            kv_bytes_per_token: 0,
            cache_capacity_tokens: 0,
            model_context_tokens: 0,
        }
    }
}
