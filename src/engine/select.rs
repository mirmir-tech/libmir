use super::{Engine, EngineInner};
use crate::{Result, RuntimeConfig};

#[cfg(all(feature = "cuda", not(feature = "metal")))]
pub(super) fn engine(config: &RuntimeConfig) -> Result<Engine> {
    Ok(Engine {
        inner: EngineInner::Cuda(cuda::CudaEngine::new_with_scheduler(
            config.cuda.clone(),
            config.kv_cache,
            config.scheduler.clone(),
        )?),
    })
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(
    clippy::unnecessary_wraps,
    reason = "all feature combinations expose one fallible engine constructor"
)]
pub(super) fn engine(config: &RuntimeConfig) -> Result<Engine> {
    let mut metal = config.metal.clone();
    metal.kv_cache = config.kv_cache;
    metal.set_max_batch_requests(config.scheduler.max_batch_requests);
    Ok(Engine {
        inner: EngineInner::Metal(metal::MetalBackend::try_new(metal)?),
    })
}

#[cfg(all(feature = "cuda", feature = "metal", target_os = "macos"))]
#[allow(
    clippy::unnecessary_wraps,
    reason = "all feature combinations expose one fallible engine constructor"
)]
pub(super) fn engine(config: &RuntimeConfig) -> Result<Engine> {
    let mut metal = config.metal.clone();
    metal.kv_cache = config.kv_cache;
    metal.set_max_batch_requests(config.scheduler.max_batch_requests);
    Ok(Engine {
        inner: EngineInner::Metal(metal::MetalBackend::try_new(metal)?),
    })
}

#[cfg(all(feature = "cuda", feature = "metal", not(target_os = "macos")))]
pub(super) fn engine(config: &RuntimeConfig) -> Result<Engine> {
    Ok(Engine {
        inner: EngineInner::Cuda(cuda::CudaEngine::new_with_scheduler(
            config.cuda.clone(),
            config.kv_cache,
            config.scheduler.clone(),
        )?),
    })
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[allow(
    clippy::unnecessary_wraps,
    reason = "all feature combinations expose one fallible engine constructor"
)]
pub(super) fn engine(_config: &RuntimeConfig) -> Result<Engine> {
    Ok(Engine { inner: EngineInner::Unavailable })
}
