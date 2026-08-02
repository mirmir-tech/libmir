/// Explicit CUDA runtime and planning policy.
use std::path::PathBuf;

use crate::backend::{CudaModelSessionConfig, CudaPlanningPolicy, CudaTuningConfig};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaConfig {
    /// CUDA device ordinal selected during backend construction.
    pub device_ordinal: usize,
    /// Bytes retained by CUDA's default memory pool after use.
    pub memory_pool_release_threshold: u64,
    /// Header roots passed explicitly to NVRTC for toolkit-backed kernels.
    pub nvrtc_include_paths: Vec<PathBuf>,
    /// Optional directory for persistent, architecture-specific PTX artifacts.
    pub nvrtc_cache_directory: Option<PathBuf>,
    /// Explicit model-level execution planning policy.
    pub planning: CudaPlanningPolicy,
    /// Persistent, shape-keyed CUDA startup tuning policy.
    pub tuning: CudaTuningConfig,
    /// Session-local prompt allocation policy.
    pub model_session: CudaModelSessionConfig,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            memory_pool_release_threshold: 512 * 1_024 * 1_024,
            nvrtc_include_paths: vec![
                PathBuf::from("/usr/local/cuda/include"),
                PathBuf::from("/usr/local/cuda/include/cccl"),
            ],
            nvrtc_cache_directory: None,
            planning: CudaPlanningPolicy::default(),
            tuning: CudaTuningConfig::default(),
            model_session: CudaModelSessionConfig::default(),
        }
    }
}
