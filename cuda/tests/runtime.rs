#[cfg(target_os = "linux")]
use libmir_cuda::{CudaConfig, CudaEngine};
#[cfg(target_os = "linux")]
use runtime::{backend::Backend, kv::CacheConfig};

#[test]
#[cfg(target_os = "linux")]
fn initializes_real_cuda_resources_and_reports_the_device() -> libmir_cuda::Result<()> {
    let engine = CudaEngine::new(CudaConfig::default(), CacheConfig::new(128))?;
    let info = engine.info();
    assert_eq!(info.name, "cuda-native");
    assert!(!info.device.is_empty());
    assert!(!info.capabilities.is_empty());
    Ok(())
}
