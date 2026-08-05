use mircuda::{
    Context, DeviceBuffer, MarlinNvFp4DenseOperands, MarlinNvFp4MoeSpec, MarlinNvFp4RepackSpec,
    MarlinNvFp4ThreadConfig, Stream, bf16,
};

use super::{CudaBackend, NvFp4Config, NvFp4WeightOnlyWeight};
use crate::{Error, Result};

mod pair;

#[derive(Debug)]
pub(in crate::backend) struct MarlinNvFp4Bf16Linear {
    weight: MarlinNvFp4Weight,
    temporary: DeviceBuffer<f32>,
    locks: DeviceBuffer<i32>,
    context: Context,
    stream: Stream,
    config: NvFp4Config,
    tokens: usize,
    atomic_reduce: bool,
}

#[derive(Clone, Debug)]
pub(super) struct MarlinNvFp4Weight {
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

impl MarlinNvFp4Bf16Linear {
    pub(super) fn supported(config: NvFp4Config) -> bool {
        config.output_features.is_multiple_of(64) && config.input_features.is_multiple_of(128)
    }

    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        source: &NvFp4WeightOnlyWeight,
    ) -> Result<Self> {
        let config = source.config;
        if !Self::supported(config) || tokens == 0 || tokens > 8 {
            return Err(Error::InvalidNvFp4("unsupported dense Marlin geometry"));
        }
        Self::from_weight(backend, tokens, config, source.marlin(backend)?, false)
    }

    fn from_weight(
        backend: &CudaBackend,
        tokens: usize,
        config: NvFp4Config,
        weight: MarlinNvFp4Weight,
        atomic_reduce: bool,
    ) -> Result<Self> {
        let sms = usize::try_from(backend.inner.device.multiprocessor_count)?;
        Ok(Self {
            weight,
            temporary: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, product(sms, 16 * 256)?)?,
            locks: backend.inner.pool.allocate_zeroed(
                &backend.inner.stream,
                usize::try_from(backend.inner.device.multiprocessor_count)? * 4,
            )?,
            context: backend.inner.context.clone(),
            stream: backend.inner.stream.clone(),
            config,
            tokens,
            atomic_reduce,
        })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        thread_config: MarlinNvFp4ThreadConfig,
    ) -> Result<()> {
        let spec = MarlinNvFp4MoeSpec::new(
            1,
            self.tokens,
            1,
            self.config.output_features,
            self.config.input_features,
            thread_config,
        )?;
        Ok(self.context.marlin_nvfp4_dense(
            &self.stream,
            spec,
            &MarlinNvFp4DenseOperands {
                input,
                weight: &self.weight.weight,
                scales: &self.weight.scales,
                global_scale: &self.weight.global_scales,
                temporary: &mut self.temporary,
                locks: &mut self.locks,
                output,
                atomic_reduce: self.atomic_reduce,
            },
        )?)
    }
}

impl NvFp4WeightOnlyWeight {
    fn marlin(&self, backend: &CudaBackend) -> Result<MarlinNvFp4Weight> {
        let mut cached = self
            .marlin
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("dense Marlin weight cache is poisoned"))?;
        if let Some(weight) = cached.as_ref() {
            return Ok(weight.clone());
        }
        let config = self.config;
        let repack = MarlinNvFp4RepackSpec::new(1, config.output_features, config.input_features)?;
        let elements = matrix_elements(config)?;
        let mut weight = backend.inner.pool.allocate(&backend.inner.stream, elements / 2)?;
        let mut scales = backend.inner.pool.allocate(&backend.inner.stream, elements / 16)?;
        let mut global_scales = backend.inner.pool.allocate(&backend.inner.stream, 1)?;
        let mut maximum = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 1)?;
        backend.inner.context.marlin_repack_nvfp4(
            &backend.inner.stream,
            repack,
            &self.weight,
            &mut weight,
        )?;
        backend.inner.context.marlin_prepare_nvfp4_scales(
            &backend.inner.stream,
            repack,
            &self.scales,
            None,
            &self.global_scale,
            &mut scales,
            &mut global_scales,
            &mut maximum,
        )?;
        let weight = MarlinNvFp4Weight { weight, scales, global_scales };
        *cached = Some(weight.clone());
        drop(cached);
        Ok(weight)
    }
}

fn matrix_elements(config: NvFp4Config) -> Result<usize> {
    product(config.input_features, config.output_features)
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::InvalidNvFp4("dense Marlin size overflow"))
}
