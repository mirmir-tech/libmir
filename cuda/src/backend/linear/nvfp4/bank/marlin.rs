use mircuda::MarlinNvFp4RepackSpec;

use super::{MarlinNvFp4Bank, NvFp4ExpertBank, NvFp4ExpertBankConfig};
use crate::{CudaBackend, Error, Result};

impl NvFp4ExpertBank {
    pub(in crate::backend) fn marlin(&self, backend: &CudaBackend) -> Result<MarlinNvFp4Bank> {
        let mut cached = self
            .marlin_single
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("Marlin bank cache lock is poisoned"))?;
        if let Some(bank) = cached.as_ref() {
            return Ok(bank.clone());
        }
        let bank = prepare(backend, self, None)?;
        *cached = Some(bank.clone());
        drop(cached);
        Ok(bank)
    }

    pub(in crate::backend) fn marlin_pair(
        &self,
        backend: &CudaBackend,
        right: &Self,
    ) -> Result<MarlinNvFp4Bank> {
        if self.config != right.config
            || self.global_values.as_ref() != right.global_values.as_ref()
        {
            return Err(Error::InvalidNvFp4("incompatible Marlin gate/up banks"));
        }
        let mut cached = self
            .marlin_pair
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("Marlin pair cache lock is poisoned"))?;
        if let Some(bank) = cached.as_ref() {
            return Ok(bank.clone());
        }
        let bank = prepare(backend, self, Some(right))?;
        *cached = Some(bank.clone());
        drop(cached);
        Ok(bank)
    }
}

fn prepare(
    backend: &CudaBackend,
    left: &NvFp4ExpertBank,
    right: Option<&NvFp4ExpertBank>,
) -> Result<MarlinNvFp4Bank> {
    let output_features = left
        .config
        .output_features
        .checked_mul(if right.is_some() {
            2
        } else {
            1
        })
        .ok_or(Error::InvalidNvFp4("Marlin output width overflow"))?;
    let config = NvFp4ExpertBankConfig { output_features, ..left.config };
    let elements = config
        .experts
        .checked_mul(config.output_features)
        .and_then(|value| value.checked_mul(config.input_features))
        .ok_or(Error::InvalidNvFp4("Marlin bank size overflow"))?;
    let mut weight = backend.inner.pool.allocate(&backend.inner.stream, elements / 2)?;
    let mut scales = backend.inner.pool.allocate(&backend.inner.stream, elements / 16)?;
    let mut global_scales = backend.inner.pool.allocate(&backend.inner.stream, config.experts)?;
    let mut maximum = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 1)?;
    let spec =
        MarlinNvFp4RepackSpec::new(config.experts, config.output_features, config.input_features)?;
    if let Some(right) = right {
        backend.inner.context.marlin_repack_nvfp4_pair(
            &backend.inner.stream,
            spec,
            &left.weight,
            &right.weight,
            &mut weight,
        )?;
    } else {
        backend.inner.context.marlin_repack_nvfp4(
            &backend.inner.stream,
            spec,
            &left.weight,
            &mut weight,
        )?;
    }
    backend.inner.context.marlin_prepare_nvfp4_scales(
        &backend.inner.stream,
        spec,
        &left.scales,
        right.map(|value| &value.scales),
        &left.global_scales,
        &mut scales,
        &mut global_scales,
        &mut maximum,
    )?;
    Ok(MarlinNvFp4Bank { weight, scales, global_scales, config })
}
