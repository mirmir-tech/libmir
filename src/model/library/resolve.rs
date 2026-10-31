//! Effective runtime configuration and engine for a model load.

use foundation::model::BackendTarget;
use runtime::kv::CacheConfig;

use super::{Library, ModelLoadOptions};
use crate::{
    Engine, Result, RuntimeConfig,
    model::{ModelDescriptor, automatic_cache, kv_sizing::KvSizing, memory_policy},
};

impl Library {
    pub(super) fn model_runtime(
        &self,
        descriptor: &ModelDescriptor,
        options: ModelLoadOptions,
    ) -> Result<(Engine, RuntimeConfig, KvSizing)> {
        let mut state = self.lock_state()?;
        let engine_config = state.model_config.as_ref().unwrap_or(&self.config).clone();
        let probe = match state.engine.take() {
            Some(engine) => engine,
            None => Engine::from_config(&engine_config)?,
        };
        let target = probe.target();
        let estimate = descriptor.memory_estimate_for(&self.config, &target);
        let memory = probe.memory_snapshot()?;
        let committed = self.memory.committed_bytes()?;
        // A per-model block count is final: no automatic sizing or resize.
        let library = options.kv_blocks.map_or_else(
            || self.config.clone(),
            |blocks| {
                let mut fixed = self.config.clone();
                fixed.kv_cache.block_count = blocks.get();
                fixed.automatic_kv_cache = false;
                fixed
            },
        );
        let mut config = automatic_cache::resolve(&library, estimate, &memory, committed);
        let resizable = probe.kv_cache_resizable(&descriptor.manifest_for(target.clone())?)?;
        let kv_sizing = KvSizing::for_load(&library, &mut config, estimate, &target, resizable);
        let resolved_estimate = descriptor.memory_estimate_for(&config, &target);
        let engine = if engine_cache_compatible(&target, config.kv_cache, engine_config.kv_cache) {
            probe
        } else {
            Engine::from_config(&config)?
        };
        tracing::info!(
            automatic = self.config.automatic_kv_cache,
            sizing = ?kv_sizing,
            blocks = config.kv_cache.block_count,
            block_size = config.kv_cache.block_size,
            capacity_tokens = u64::from(config.kv_cache.block_count)
                .saturating_mul(u64::try_from(config.kv_cache.block_size).unwrap_or(u64::MAX)),
            kv_bytes_per_token = estimate.kv_bytes_per_token,
            kv_cache_bytes = resolved_estimate.kv_cache_bytes,
            platform_reserve_bytes = memory_policy::platform_reserve(config.memory, &memory),
            transient_reserve_bytes = memory_policy::transient_reserve(resolved_estimate, &memory),
            planned_residency_bytes = memory_policy::planned_residency(resolved_estimate, &memory),
            memory_source = memory.source,
            "resolved model KV cache"
        );
        state.model_config = Some(config.clone());
        state.engine = Some(engine.clone());
        drop(state);
        Ok((engine, config, kv_sizing))
    }
}

pub(super) fn engine_cache_compatible(
    target: &BackendTarget,
    resolved: CacheConfig,
    current: CacheConfig,
) -> bool {
    resolved == current
        || (*target == BackendTarget::Metal
            && resolved.block_size == current.block_size
            && resolved.dtype == current.dtype)
}
