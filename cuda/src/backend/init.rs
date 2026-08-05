use std::sync::Arc;

use mircuda::{Driver, MemoryPoolStats};

use super::{
    CudaBackend, CudaExecutionPlanner, CudaHardwareProfile, CudaRuntime, tuning::CudaAutoTuner,
};
use crate::{CudaConfig, Error, Result, TensorUploadBatch, kernels::DenseCast};

impl CudaBackend {
    /// Initializes one CUDA device, explicit stream, memory pool, and plan
    /// cache.
    pub fn new(config: CudaConfig) -> Result<Self> {
        let CudaConfig {
            device_ordinal,
            memory_pool_release_threshold,
            nvrtc_include_paths,
            nvrtc_cache_directory,
            planning,
            tuning,
            model_session: _,
        } = config;
        let driver = Driver::initialize()?;
        let device = driver
            .devices()?
            .into_iter()
            .find(|device| device.ordinal() == device_ordinal)
            .ok_or(Error::DeviceUnavailable(device_ordinal))?;
        let context = driver.create_context(device)?;
        let info = context.device_info()?;
        let stream = context.create_stream()?;
        let auxiliary_stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        pool.set_release_threshold(memory_pool_release_threshold)?;
        let compiler = Arc::new(mircuda::Compiler::with_config(
            context.clone(),
            mircuda::CompilerConfig {
                include_paths: nvrtc_include_paths,
                cache_directory: nvrtc_cache_directory,
            },
        )?);
        let planner = CudaExecutionPlanner::new(CudaHardwareProfile::from_device(&info)?, planning);
        let tuner = CudaAutoTuner::new(&info, tuning);
        Ok(Self {
            inner: Arc::new(CudaRuntime {
                device: info,
                context,
                stream,
                auxiliary_stream,
                pool,
                compiler,
                mxfp8_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
                nvfp4_bucket_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
                nvfp4_marlin_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
                planner,
                tuner,
            }),
        })
    }

    /// Returns the selected CUDA device properties.
    #[must_use]
    pub fn device_info(&self) -> &mircuda::DeviceInfo {
        &self.inner.device
    }

    /// Returns the immutable model-level execution planner.
    #[must_use]
    pub fn execution_planner(&self) -> &CudaExecutionPlanner {
        &self.inner.planner
    }

    /// Returns CUDA pool accounting without synchronizing execution.
    pub fn memory_pool_stats(&self) -> Result<MemoryPoolStats> {
        Ok(self.inner.pool.stats()?)
    }

    /// Returns driver-reported free and total device memory.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        Ok(self.inner.context.memory_info()?)
    }

    /// Starts a direct-from-checkpoint asynchronous tensor upload batch.
    #[must_use]
    pub fn begin_tensor_upload(&self) -> TensorUploadBatch {
        TensorUploadBatch::new(
            self.inner.context.clone(),
            self.inner.stream.clone(),
            self.inner.pool.clone(),
        )
    }

    pub(crate) fn prepare_dense_cast(&self) -> Result<DenseCast> {
        DenseCast::compile(&self.inner.compiler)
    }

    pub(crate) fn auxiliary_backend(&self) -> Self {
        Self {
            inner: Arc::new(CudaRuntime {
                device: self.inner.device.clone(),
                context: self.inner.context.clone(),
                stream: self.inner.auxiliary_stream.clone(),
                auxiliary_stream: self.inner.auxiliary_stream.clone(),
                pool: self.inner.pool.clone(),
                compiler: self.inner.compiler.clone(),
                mxfp8_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
                nvfp4_bucket_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
                nvfp4_marlin_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
                planner: self.inner.planner,
                tuner: self.inner.tuner.clone(),
            }),
        }
    }
}
