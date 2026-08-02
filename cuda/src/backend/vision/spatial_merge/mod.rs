mod input;
mod layer;
mod primitives;
mod runner;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, bf16};
use models::{layout::SpatialMergeVisionConfig, vision::SpatialMergePreprocessedImage};

use super::super::CudaBackend;
use crate::{CudaTensorSet, Error, Result};

#[derive(Debug)]
pub struct CudaSpatialMergeVisionOutput {
    pub(crate) hidden: DeviceBuffer<bf16>,
    pub(crate) tokens: usize,
    pub(crate) width: usize,
    _lease: SpatialRunnerLease,
}

#[derive(Debug)]
pub struct CudaSpatialMergeVisionTower {
    backend: CudaBackend,
    config: SpatialMergeVisionConfig,
    tensors: CudaTensorSet,
    runners: Arc<Mutex<SpatialRunnerPool>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SpatialRunnerGeometry {
    grid_height: usize,
    grid_width: usize,
}

#[derive(Debug, Default)]
struct SpatialRunnerPool {
    available: Vec<(SpatialRunnerGeometry, runner::SpatialMergeRunner)>,
    created: usize,
}

const CACHED_RUNNERS: usize = 1;

#[derive(Debug)]
struct SpatialRunnerLease {
    geometry: SpatialRunnerGeometry,
    runner: Option<runner::SpatialMergeRunner>,
    pool: Arc<Mutex<SpatialRunnerPool>>,
}

impl CudaSpatialMergeVisionTower {
    pub(crate) fn new(
        backend: &CudaBackend,
        config: SpatialMergeVisionConfig,
        tensors: CudaTensorSet,
    ) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tensors,
            runners: Arc::new(Mutex::new(SpatialRunnerPool::default())),
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn forward_preprocessed(
        &self,
        image: &SpatialMergePreprocessedImage,
    ) -> Result<CudaSpatialMergeVisionOutput> {
        let mut lease = self.checkout(image)?;
        lease.runner_mut()?.execute()?;
        let (hidden, tokens, width) = lease.runner()?.output();
        Ok(CudaSpatialMergeVisionOutput { hidden, tokens, width, _lease: lease })
    }

    pub(crate) fn forward_preprocessed_scheduled<F>(
        &self,
        image: &SpatialMergePreprocessedImage,
        schedule: &mut F,
    ) -> Result<CudaSpatialMergeVisionOutput>
    where
        F: FnMut(&mut dyn FnMut() -> Result<()>) -> Result<()>,
    {
        let mut lease = None;
        schedule(&mut || {
            lease = Some(self.checkout(image)?);
            Ok(())
        })?;
        let mut lease =
            lease.ok_or_else(|| Error::State("spatial-merge vision step was skipped".into()))?;
        schedule(&mut || lease.runner_mut()?.execute_input())?;
        let layers = lease.runner()?.layer_count();
        for index in 0..layers {
            schedule(&mut || lease.runner_mut()?.execute_layer(index))?;
        }
        schedule(&mut || lease.runner_mut()?.execute_merger())?;
        let (hidden, tokens, width) = lease.runner()?.output();
        Ok(CudaSpatialMergeVisionOutput { hidden, tokens, width, _lease: lease })
    }

    pub(crate) const fn layer_count(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn checkout(&self, image: &SpatialMergePreprocessedImage) -> Result<SpatialRunnerLease> {
        let geometry = SpatialRunnerGeometry {
            grid_height: image.grid_height,
            grid_width: image.grid_width,
        };
        let cached = {
            let Ok(mut pool) = self.runners.lock() else {
                return Err(Error::State("spatial-merge runner pool is poisoned".into()));
            };
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
                runner::SpatialMergeRunner::new(&self.backend, &self.config, &self.tensors, image)?;
            let Ok(mut pool) = self.runners.lock() else {
                return Err(Error::State("spatial-merge runner pool is poisoned".into()));
            };
            pool.created += 1;
            drop(pool);
            runner
        };
        Ok(SpatialRunnerLease {
            geometry,
            runner: Some(runner),
            pool: self.runners.clone(),
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    fn runner_pool_stats(&self) -> Result<(usize, usize)> {
        let Ok(pool) = self.runners.lock() else {
            return Err(Error::State("spatial-merge runner pool is poisoned".into()));
        };
        Ok((pool.created, pool.available.len()))
    }
}

impl SpatialRunnerLease {
    fn runner(&self) -> Result<&runner::SpatialMergeRunner> {
        self.runner
            .as_ref()
            .ok_or_else(|| Error::State("spatial-merge runner lease is empty".into()))
    }

    fn runner_mut(&mut self) -> Result<&mut runner::SpatialMergeRunner> {
        self.runner
            .as_mut()
            .ok_or_else(|| Error::State("spatial-merge runner lease is empty".into()))
    }
}

impl Drop for SpatialRunnerLease {
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

fn validate_config(config: &SpatialMergeVisionConfig) -> Result<()> {
    let head_dim = config.hidden_size.checked_div(config.num_attention_heads).unwrap_or_default();
    if config.hidden_size == 0
        || config.num_attention_heads == 0
        || !config.hidden_size.is_multiple_of(config.num_attention_heads)
        || !head_dim.is_multiple_of(8)
        || head_dim > 256
        || config.in_channels != 3
        || config.spatial_merge_size == 0
        || config.hidden_activation != "gelu_pytorch_tanh"
    {
        return Err(Error::UnsupportedVisionContract(format!(
            "unsupported spatial-merge vision geometry or activation: hidden={}, heads={}, merge={}, activation={}",
            config.hidden_size,
            config.num_attention_heads,
            config.spatial_merge_size,
            config.hidden_activation
        )));
    }
    Ok(())
}
