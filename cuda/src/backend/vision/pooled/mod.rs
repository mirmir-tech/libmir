mod input;
mod layer;
mod runner;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, bf16};
use models::{layout::PooledVisionConfig, vision::PooledPreprocessedImage};

use super::super::CudaBackend;
use crate::{CudaTensorSet, Error, Result};

#[derive(Debug)]
pub struct CudaPooledVisionOutput {
    pub(crate) hidden: DeviceBuffer<bf16>,
    pub(crate) tokens: usize,
    pub(crate) width: usize,
    _lease: PooledRunnerLease,
}

#[derive(Debug)]
pub struct CudaPooledVisionTower {
    backend: CudaBackend,
    config: PooledVisionConfig,
    tensors: CudaTensorSet,
    runners: Arc<Mutex<PooledRunnerPool>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PooledRunnerGeometry {
    grid_height: usize,
    grid_width: usize,
}

#[derive(Debug, Default)]
struct PooledRunnerPool {
    available: Vec<(PooledRunnerGeometry, runner::PooledRunner)>,
    created: usize,
}

const CACHED_RUNNERS: usize = 1;

#[derive(Debug)]
struct PooledRunnerLease {
    geometry: PooledRunnerGeometry,
    runner: Option<runner::PooledRunner>,
    pool: Arc<Mutex<PooledRunnerPool>>,
}

impl CudaPooledVisionTower {
    pub(crate) fn new(
        backend: &CudaBackend,
        config: PooledVisionConfig,
        tensors: CudaTensorSet,
    ) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tensors,
            runners: Arc::new(Mutex::new(PooledRunnerPool::default())),
        })
    }

    pub(crate) fn forward_preprocessed_scheduled<F>(
        &self,
        image: &PooledPreprocessedImage,
        schedule: &mut F,
    ) -> Result<CudaPooledVisionOutput>
    where
        F: FnMut(&mut dyn FnMut() -> Result<()>) -> Result<()>,
    {
        let mut lease = None;
        schedule(&mut || {
            lease = Some(self.checkout(image)?);
            Ok(())
        })?;
        let mut lease =
            lease.ok_or_else(|| Error::State("pooled vision step was skipped".into()))?;
        schedule(&mut || lease.runner_mut()?.execute_input())?;
        let layers = lease.runner()?.layer_count();
        for index in 0..layers {
            schedule(&mut || lease.runner_mut()?.execute_layer(index))?;
        }
        schedule(&mut || lease.runner_mut()?.execute_output())?;
        let (hidden, tokens, width) = lease.runner()?.output();
        Ok(CudaPooledVisionOutput { hidden, tokens, width, _lease: lease })
    }

    pub(crate) const fn layer_count(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub(crate) const fn bidirectional_image_attention(&self) -> bool {
        self.config.bidirectional_image_attention
    }

    fn checkout(&self, image: &PooledPreprocessedImage) -> Result<PooledRunnerLease> {
        let geometry = PooledRunnerGeometry {
            grid_height: image.grid_height,
            grid_width: image.grid_width,
        };
        let cached = {
            let mut pool = self
                .runners
                .lock()
                .map_err(|_| Error::State("pooled runner pool is poisoned".into()))?;
            pool.available
                .iter()
                .rposition(|(candidate, _runner)| *candidate == geometry)
                .map(|index| pool.available.swap_remove(index).1)
        };
        let runner = if let Some(mut runner) = cached {
            runner.update_input(image)?;
            runner
        } else {
            let runner =
                runner::PooledRunner::new(&self.backend, &self.config, &self.tensors, image)?;
            self.runners
                .lock()
                .map_err(|_| Error::State("pooled runner pool is poisoned".into()))?
                .created += 1;
            runner
        };
        Ok(PooledRunnerLease {
            geometry,
            runner: Some(runner),
            pool: self.runners.clone(),
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    fn runner_pool_stats(&self) -> Result<(usize, usize)> {
        let pool = self
            .runners
            .lock()
            .map_err(|_| Error::State("pooled runner pool is poisoned".into()))?;
        Ok((pool.created, pool.available.len()))
    }
}

impl PooledRunnerLease {
    fn runner(&self) -> Result<&runner::PooledRunner> {
        self.runner
            .as_ref()
            .ok_or_else(|| Error::State("pooled runner lease is empty".into()))
    }

    fn runner_mut(&mut self) -> Result<&mut runner::PooledRunner> {
        self.runner
            .as_mut()
            .ok_or_else(|| Error::State("pooled runner lease is empty".into()))
    }
}

impl Drop for PooledRunnerLease {
    fn drop(&mut self) {
        let Some(runner) = self.runner.take() else {
            return;
        };
        if let Ok(mut pool) = self.pool.lock() {
            let evicted =
                (pool.available.len() == CACHED_RUNNERS).then(|| pool.available.remove(0).1);
            pool.available.push((self.geometry, runner));
            drop(pool);
            drop(evicted);
        }
    }
}

fn validate_config(config: &PooledVisionConfig) -> Result<()> {
    let projected = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or(Error::InvalidVisionKernel("pooled hidden width overflow"))?;
    if config.hidden_size != projected
        || config.num_key_value_heads == 0
        || !config.num_attention_heads.is_multiple_of(config.num_key_value_heads)
        || config.pooling_kernel_size == 0
        || config.hidden_activation != "gelu_pytorch_tanh"
        || !config.rms_norm_eps.is_finite()
        || config.rms_norm_eps < 0.0
    {
        return Err(Error::UnsupportedVisionContract(format!(
            "unsupported pooled vision geometry or activation: hidden={}, heads={}, kv_heads={}, head_dim={}, pool={}, activation={}",
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.pooling_kernel_size,
            config.hidden_activation
        )));
    }
    Ok(())
}
