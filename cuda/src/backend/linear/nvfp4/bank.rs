use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use mircuda::DeviceBuffer;
use models::weights::TensorInfo;

use super::CudaBackend;
use crate::{
    Error, Result,
    kernels::{BankScaleGeometry, NvFp4GroupedPreparation, scale_elements},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4ExpertBankConfig {
    pub experts: usize,
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Clone, Copy)]
pub struct NvFp4ExpertSource<'a> {
    pub weight: &'a TensorInfo,
    pub weight_scale: &'a TensorInfo,
    pub weight_scale_2: &'a TensorInfo,
    pub input_scale: &'a TensorInfo,
}

/// Reusable shared handles to one contiguous device-resident expert bank.
///
/// Cloning the bank does not upload or copy its device tensors.
#[derive(Clone, Debug)]
pub struct NvFp4ExpertBank {
    pub(super) weight: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
    pub(super) cutlass_scales: DeviceBuffer<u8>,
    pub(super) global_scales: DeviceBuffer<f32>,
    pub(super) input_scales: DeviceBuffer<f32>,
    pub(super) combined_scales: DeviceBuffer<f32>,
    pub(super) config: NvFp4ExpertBankConfig,
}

impl CudaBackend {
    /// Uploads separate expert tensors into one contiguous device-side bank.
    pub fn prepare_nvfp4_expert_bank(
        &self,
        config: NvFp4ExpertBankConfig,
        sources: &[NvFp4ExpertSource<'_>],
    ) -> Result<NvFp4ExpertBank> {
        NvFp4ExpertBank::load(self, config, sources)
    }
}

impl NvFp4ExpertBank {
    #[must_use]
    pub const fn config(&self) -> NvFp4ExpertBankConfig {
        self.config
    }

    fn load(
        backend: &CudaBackend,
        config: NvFp4ExpertBankConfig,
        sources: &[NvFp4ExpertSource<'_>],
    ) -> Result<Self> {
        validate(config, sources)?;
        let weight_bytes = config.output_features * config.input_features / 2;
        let scale_bytes = config.output_features * config.input_features / 16;
        let weight = upload_bytes(backend, sources, weight_bytes, |source| source.weight)?;
        let scales = upload_bytes(backend, sources, scale_bytes, |source| source.weight_scale)?;
        let cutlass_scale_elements = config
            .experts
            .checked_mul(scale_elements(config.output_features, config.input_features)?)
            .ok_or(Error::InvalidNvFp4("expert CUTLASS scale size overflow"))?;
        let mut cutlass_scales = backend
            .inner
            .pool
            .allocate_zeroed::<u8>(&backend.inner.stream, cutlass_scale_elements)?;
        NvFp4GroupedPreparation::compile(&backend.inner.compiler)?.prepare_bank_scales(
            &backend.inner.stream,
            &scales,
            &mut cutlass_scales,
            BankScaleGeometry {
                experts: config.experts,
                rows: config.output_features,
                columns: config.input_features,
            },
        )?;
        let global_values = sources
            .iter()
            .map(|source| read_scalar(source.weight_scale_2))
            .collect::<Result<Vec<_>>>()?;
        let global_scales = upload_scalars(backend, &global_values)?;
        let input_values = sources
            .iter()
            .map(|source| read_scalar(source.input_scale))
            .collect::<Result<Vec<_>>>()?;
        let input_scales = upload_scalars(backend, &input_values)?;
        let combined_values = input_values
            .iter()
            .zip(&global_values)
            .map(|(input, weight)| input * weight)
            .collect::<Vec<_>>();
        let combined_scales = upload_scalars(backend, &combined_values)?;
        backend.inner.stream.synchronize()?;
        Ok(Self {
            weight,
            scales,
            cutlass_scales,
            global_scales,
            input_scales,
            combined_scales,
            config,
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn synthetic_zero(
        backend: &CudaBackend,
        config: NvFp4ExpertBankConfig,
    ) -> Result<Self> {
        let matrix = config
            .input_features
            .checked_mul(config.output_features)
            .and_then(|value| value.checked_mul(config.experts))
            .ok_or(Error::InvalidNvFp4("synthetic expert bank size overflow"))?;
        let scale_count = config
            .experts
            .checked_mul(scale_elements(config.output_features, config.input_features)?)
            .ok_or(Error::InvalidNvFp4("synthetic expert scale size overflow"))?;
        let weight = backend.inner.pool.allocate_zeroed(&backend.inner.stream, matrix / 2)?;
        let scales = backend.inner.pool.allocate_zeroed(&backend.inner.stream, matrix / 16)?;
        let mut cutlass_scales =
            backend.inner.pool.allocate_zeroed(&backend.inner.stream, scale_count)?;
        NvFp4GroupedPreparation::compile(&backend.inner.compiler)?.prepare_bank_scales(
            &backend.inner.stream,
            &scales,
            &mut cutlass_scales,
            BankScaleGeometry {
                experts: config.experts,
                rows: config.output_features,
                columns: config.input_features,
            },
        )?;
        let zero_scales =
            || backend.inner.pool.allocate_zeroed(&backend.inner.stream, config.experts);
        let bank = Self {
            weight,
            scales,
            cutlass_scales,
            global_scales: zero_scales()?,
            input_scales: zero_scales()?,
            combined_scales: zero_scales()?,
            config,
        };
        backend.inner.stream.synchronize()?;
        Ok(bank)
    }
}

fn upload_scalars(backend: &CudaBackend, values: &[f32]) -> Result<DeviceBuffer<f32>> {
    let mut staging = backend.inner.context.allocate_pinned::<f32>(values.len())?;
    staging.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<f32>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut staging, &mut device)?;
    Ok(device)
}

fn upload_bytes<'a>(
    backend: &CudaBackend,
    sources: &'a [NvFp4ExpertSource<'a>],
    bytes_per_expert: usize,
    tensor: impl Fn(&'a NvFp4ExpertSource<'a>) -> &'a TensorInfo,
) -> Result<DeviceBuffer<u8>> {
    let total = sources
        .len()
        .checked_mul(bytes_per_expert)
        .ok_or(Error::InvalidNvFp4("expert bank size overflow"))?;
    let mut staging = backend.inner.context.allocate_pinned::<u8>(total)?;
    let read = staging.with_bytes_mut(|target| -> Result<()> {
        for (expert, source) in sources.iter().enumerate() {
            let info = tensor(source);
            let start = expert * bytes_per_expert;
            let mut file = File::open(&info.file)?;
            file.seek(SeekFrom::Start(info.payload_start()?))?;
            file.read_exact(&mut target[start..start + bytes_per_expert])?;
        }
        Ok(())
    })?;
    read?;
    let mut device = backend.inner.pool.allocate::<u8>(&backend.inner.stream, total)?;
    backend.inner.stream.copy_to_device(&mut staging, &mut device)?;
    backend.inner.stream.synchronize()?;
    Ok(device)
}

fn read_scalar(info: &TensorInfo) -> Result<f32> {
    let mut file = File::open(&info.file)?;
    file.seek(SeekFrom::Start(info.payload_start()?))?;
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn validate(config: NvFp4ExpertBankConfig, sources: &[NvFp4ExpertSource<'_>]) -> Result<()> {
    if config.experts == 0
        || config.input_features == 0
        || config.output_features == 0
        || !config.input_features.is_multiple_of(16)
        || sources.len() != config.experts
    {
        return Err(Error::InvalidNvFp4("invalid expert bank geometry"));
    }
    let weight_shape = [config.output_features, config.input_features / 2];
    let scale_shape = [config.output_features, config.input_features / 16];
    for source in sources {
        validate_tensor(source.weight, "U8", &weight_shape)?;
        validate_tensor(source.weight_scale, "F8_E4M3", &scale_shape)?;
        validate_tensor(source.weight_scale_2, "F32", &[])?;
        validate_tensor(source.input_scale, "F32", &[])?;
    }
    Ok(())
}

fn validate_tensor(info: &TensorInfo, dtype: &str, shape: &[usize]) -> Result<()> {
    if info.dtype != dtype || info.shape != shape {
        return Err(Error::InvalidNvFp4("expert bank tensor metadata mismatch"));
    }
    Ok(())
}
