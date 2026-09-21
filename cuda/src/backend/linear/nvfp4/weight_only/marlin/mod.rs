use mircuda::{
    Context, DeviceBuffer, MarlinNvFp4DenseOperands, MarlinNvFp4MoeSpec, MarlinNvFp4RepackSpec,
    MarlinNvFp4ThreadConfig, Stream, bf16,
};

use super::{CudaBackend, NvFp4Config, NvFp4WeightOnlyWeight};
use crate::{Error, Result};

mod pair;

/// The kernel serves at most one sixteen-row block per launch; longer batches
/// run as consecutive groups of rows over the same weights.
const GROUP_ROWS: usize = 16;
pub(in crate::backend) const MAX_DENSE_MARLIN_TOKENS: usize = 64;

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
        if !Self::supported(config) || tokens == 0 || tokens > MAX_DENSE_MARLIN_TOKENS {
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

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "row groups are written through views of the exclusively borrowed output"
    )]
    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        thread_config: MarlinNvFp4ThreadConfig,
    ) -> Result<()> {
        let (inputs, outputs) = (self.config.input_features, self.config.output_features);
        let mut row = 0;
        while row < self.tokens {
            let rows = (self.tokens - row).min(GROUP_ROWS);
            let spec = MarlinNvFp4MoeSpec::new(1, rows, 1, outputs, inputs, thread_config)?;
            let group_input = input.slice(product(row, inputs)?..product(row + rows, inputs)?)?;
            let mut group_output =
                output.slice(product(row, outputs)?..product(row + rows, outputs)?)?;
            self.context.marlin_nvfp4_dense(
                &self.stream,
                spec,
                &MarlinNvFp4DenseOperands {
                    input: &group_input,
                    weight: &self.weight.weight,
                    scales: &self.weight.scales,
                    global_scale: &self.weight.global_scales,
                    temporary: &mut self.temporary,
                    locks: &mut self.locks,
                    output: &mut group_output,
                    atomic_reduce: self.atomic_reduce,
                },
            )?;
            row += rows;
        }
        Ok(())
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

#[cfg(test)]
mod tests;
