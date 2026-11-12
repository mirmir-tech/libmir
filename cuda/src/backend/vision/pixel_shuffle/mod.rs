//! The pixel-shuffle vision encoder on CUDA: prefix tokens and patches
//! with resized absolute positions through pre-norm transformer blocks,
//! then the pixel shuffle and the RMS-normalized squared-rectifier projector
//! into the decoder's hidden size. Runs scheduled step by step, so decodes of
//! other sessions interleave with a long image.

mod layer;
mod projector;
mod runner;
mod scratch;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, bf16};
use models::{layout::PixelShuffleVisionConfig, vision::PixelShufflePreprocessedImage};

use super::super::CudaBackend;
use crate::{CudaTensorSet, Error, Result, kernels::PixelShuffleGeometry};

#[derive(Debug)]
pub struct CudaPixelShuffleVisionOutput {
    pub(crate) hidden: DeviceBuffer<bf16>,
    pub(crate) tokens: usize,
    pub(crate) width: usize,
    _lease: RunnerLease,
}

#[derive(Debug)]
pub struct CudaPixelShuffleVisionTower {
    backend: CudaBackend,
    config: PixelShuffleVisionConfig,
    tensors: CudaTensorSet,
    runners: Arc<Mutex<RunnerPool>>,
}

/// Runners kept for reuse: one, the last image's grid.
const CACHED_RUNNERS: usize = 1;

#[derive(Debug, Default)]
struct RunnerPool {
    available: Vec<(PixelShuffleGeometry, runner::PixelShuffleRunner)>,
    created: usize,
}

#[derive(Debug)]
struct RunnerLease {
    geometry: PixelShuffleGeometry,
    runner: Option<runner::PixelShuffleRunner>,
    pool: Arc<Mutex<RunnerPool>>,
}

impl CudaPixelShuffleVisionTower {
    pub(crate) fn new(
        backend: &CudaBackend,
        config: PixelShuffleVisionConfig,
        tensors: CudaTensorSet,
    ) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tensors,
            runners: Arc::new(Mutex::new(RunnerPool::default())),
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn forward_preprocessed(
        &self,
        image: &PixelShufflePreprocessedImage,
    ) -> Result<CudaPixelShuffleVisionOutput> {
        let mut lease = self.checkout(image)?;
        lease.runner_mut()?.execute()?;
        let (hidden, tokens, width) = lease.runner()?.output()?;
        Ok(CudaPixelShuffleVisionOutput { hidden, tokens, width, _lease: lease })
    }

    pub(crate) fn forward_preprocessed_scheduled<F>(
        &self,
        image: &PixelShufflePreprocessedImage,
        schedule: &mut F,
    ) -> Result<CudaPixelShuffleVisionOutput>
    where
        F: FnMut(&mut dyn FnMut() -> Result<()>) -> Result<()>,
    {
        let mut lease = None;
        schedule(&mut || {
            lease = Some(self.checkout(image)?);
            Ok(())
        })?;
        let mut lease =
            lease.ok_or_else(|| Error::State("pixel-shuffle vision step was skipped".into()))?;
        schedule(&mut || lease.runner_mut()?.execute_input())?;
        for index in 0..lease.runner()?.layer_count() {
            schedule(&mut || lease.runner_mut()?.execute_layer(index))?;
        }
        schedule(&mut || lease.runner_mut()?.execute_projector())?;
        let (hidden, tokens, width) = lease.runner()?.output()?;
        Ok(CudaPixelShuffleVisionOutput { hidden, tokens, width, _lease: lease })
    }

    pub(crate) const fn layer_count(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn checkout(&self, image: &PixelShufflePreprocessedImage) -> Result<RunnerLease> {
        let geometry = runner::geometry(&self.config, image);
        let cached = {
            let mut pool = self.pool()?;
            (pool.available.iter())
                .rposition(|(candidate, _)| *candidate == geometry)
                .map(|index| pool.available.swap_remove(index).1)
        };
        let runner = if let Some(mut runner) = cached {
            runner.update_input(image)?;
            runner
        } else {
            let runner =
                runner::PixelShuffleRunner::new(&self.backend, &self.config, &self.tensors, image)?;
            self.pool()?.created += 1;
            runner
        };
        Ok(RunnerLease {
            geometry,
            runner: Some(runner),
            pool: self.runners.clone(),
        })
    }

    fn pool(&self) -> Result<std::sync::MutexGuard<'_, RunnerPool>> {
        self.runners
            .lock()
            .map_err(|_| Error::State("pixel-shuffle runner pool is poisoned".into()))
    }

    #[cfg(all(test, target_os = "linux"))]
    fn runner_pool_stats(&self) -> Result<(usize, usize)> {
        let pool = self.pool()?;
        Ok((pool.created, pool.available.len()))
    }
}

impl RunnerLease {
    fn runner(&self) -> Result<&runner::PixelShuffleRunner> {
        self.runner
            .as_ref()
            .ok_or_else(|| Error::State("pixel-shuffle runner lease is empty".into()))
    }

    fn runner_mut(&mut self) -> Result<&mut runner::PixelShuffleRunner> {
        self.runner
            .as_mut()
            .ok_or_else(|| Error::State("pixel-shuffle runner lease is empty".into()))
    }
}

impl Drop for RunnerLease {
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

fn validate_config(config: &PixelShuffleVisionConfig) -> Result<()> {
    let heads = config.num_attention_heads;
    let head_dim = config.hidden_size.checked_div(heads).unwrap_or_default();
    if config.hidden_size == 0
        || heads == 0
        || !config.hidden_size.is_multiple_of(heads)
        || !head_dim.is_multiple_of(8)
        || head_dim > 256
        || config.num_channels != 3
        || config.shuffle_factor == 0
        || config.position_grid == 0
        || config.hidden_activation != "gelu"
    {
        return Err(Error::UnsupportedVisionContract(format!(
            "unsupported pixel-shuffle vision geometry or activation: hidden={}, heads={heads}, factor={}, activation={}",
            config.hidden_size, config.shuffle_factor, config.hidden_activation
        )));
    }
    Ok(())
}
