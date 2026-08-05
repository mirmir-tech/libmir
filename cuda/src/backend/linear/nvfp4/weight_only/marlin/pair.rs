use mircuda::{DeviceBuffer, MarlinNvFp4RepackSpec};

use super::{MarlinNvFp4Bf16Linear, MarlinNvFp4Weight};
use crate::{CudaBackend, Error, NvFp4Config, NvFp4WeightOnlyWeight, Result};

impl MarlinNvFp4Bf16Linear {
    pub(in crate::backend) fn new_pair(
        backend: &CudaBackend,
        tokens: usize,
        left: &NvFp4WeightOnlyWeight,
        right: &NvFp4WeightOnlyWeight,
    ) -> Result<Option<Self>> {
        if left.config != right.config || tokens == 0 || tokens > 8 {
            return Ok(None);
        }
        let output = left
            .config
            .output_features
            .checked_mul(2)
            .ok_or(Error::InvalidNvFp4("Marlin pair output width overflow"))?;
        let config = NvFp4Config::new(left.config.input_features, output);
        if !Self::supported(config)
            || read_scalar(backend, &left.global_scale)?.to_bits()
                != read_scalar(backend, &right.global_scale)?.to_bits()
        {
            return Ok(None);
        }
        let weight = left.marlin_pair(backend, right, config)?;
        Self::from_weight(backend, tokens, config, weight, false).map(Some)
    }
}

impl NvFp4WeightOnlyWeight {
    fn marlin_pair(
        &self,
        backend: &CudaBackend,
        right: &Self,
        config: NvFp4Config,
    ) -> Result<MarlinNvFp4Weight> {
        let mut cached = self.marlin_pair.lock().map_err(|_| {
            Error::InvalidExecutionPlan("dense Marlin pair weight cache is poisoned")
        })?;
        if let Some(weight) = cached.as_ref() {
            return Ok(weight.clone());
        }
        let repack = MarlinNvFp4RepackSpec::new(1, config.output_features, config.input_features)?;
        let elements = config
            .input_features
            .checked_mul(config.output_features)
            .ok_or(Error::InvalidNvFp4("dense Marlin pair size overflow"))?;
        let mut weight = backend.inner.pool.allocate(&backend.inner.stream, elements / 2)?;
        let mut scales = backend.inner.pool.allocate(&backend.inner.stream, elements / 16)?;
        let mut global_scales = backend.inner.pool.allocate(&backend.inner.stream, 1)?;
        let mut maximum = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 1)?;
        backend.inner.context.marlin_repack_nvfp4_pair(
            &backend.inner.stream,
            repack,
            &self.weight,
            &right.weight,
            &mut weight,
        )?;
        backend.inner.context.marlin_prepare_nvfp4_scales(
            &backend.inner.stream,
            repack,
            &self.scales,
            Some(&right.scales),
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

fn read_scalar(backend: &CudaBackend, source: &DeviceBuffer<f32>) -> Result<f32> {
    let mut host = backend.inner.context.allocate_pinned::<f32>(1)?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    host.to_vec()?.first().copied().ok_or_else(|| Error::InvalidTensorSize {
        name: "NVFP4 global scale".into(),
        expected: 1,
        actual: 0,
    })
}
