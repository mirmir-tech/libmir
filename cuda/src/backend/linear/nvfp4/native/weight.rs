use mircuda::DeviceBuffer;

use super::super::{
    CudaBackend, NvFp4Config, NvFp4Tensors, f32_tensor, u8_tensor, validate_tensors,
};
use crate::{
    Error, Result,
    kernels::{NvFp4Preparation, NvFp4Spec},
};

#[derive(Clone, Debug)]
pub(in crate::backend::linear::nvfp4) struct NativeNvFp4Weight {
    pub(super) weight: DeviceBuffer<u8>,
    pub(super) weight_scales: DeviceBuffer<u8>,
    pub(super) input_scale: DeviceBuffer<f32>,
    pub(super) input_global: f32,
    pub(super) alpha: f32,
    pub(in crate::backend::linear::nvfp4) config: NvFp4Config,
}

impl NativeNvFp4Weight {
    pub(in crate::backend::linear::nvfp4) fn prepare(
        backend: &CudaBackend,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<Self> {
        let spec = NvFp4Spec::new(config.input_features, config.output_features)?;
        validate_tensors(spec, tensors)?;
        let preparation = NvFp4Preparation::compile(&backend.inner.compiler)?;
        let mut weight =
            backend.inner.pool.allocate::<u8>(&backend.inner.stream, spec.elements()? / 2)?;
        let mut weight_scales = backend
            .inner
            .pool
            .allocate_zeroed::<u8>(&backend.inner.stream, spec.scale_elements()?)?;
        preparation.prepare_weight(
            &backend.inner.stream,
            spec,
            u8_tensor(tensors.weight, "U8")?,
            u8_tensor(tensors.weight_scale, "F8_E4M3")?,
            &mut weight,
            &mut weight_scales,
        )?;
        let input_global = tensors
            .scale_mode
            .multiplier(read_scalar(backend, f32_tensor(tensors.input_scale)?)?)?;
        let weight_global = tensors
            .scale_mode
            .multiplier(read_scalar(backend, f32_tensor(tensors.weight_scale_2)?)?)?;
        Ok(Self {
            weight,
            weight_scales,
            input_scale: copy_scalar(backend, input_global)?,
            input_global,
            alpha: input_global * weight_global,
            config,
        })
    }

    pub(in crate::backend::linear::nvfp4) fn pack<const N: usize>(
        backend: &CudaBackend,
        weights: [&Self; N],
    ) -> Result<Self> {
        let first = weights.first().ok_or(Error::InvalidNvFp4("projection pack is empty"))?;
        let input_features = first.config.input_features;
        if weights.iter().any(|weight| {
            weight.config.input_features != input_features
                || !weight.config.output_features.is_multiple_of(128)
                || weight.input_global.to_bits() != first.input_global.to_bits()
                || weight.alpha.to_bits() != first.alpha.to_bits()
        }) {
            return Err(Error::InvalidNvFp4("projection pack geometry or scale differs"));
        }
        let output_features = weights.iter().try_fold(0_usize, |total, weight| {
            total
                .checked_add(weight.config.output_features)
                .ok_or(Error::InvalidNvFp4("projection pack output overflow"))
        })?;
        let spec = NvFp4Spec::new(input_features, output_features)?;
        let mut packed_weight =
            backend.inner.pool.allocate::<u8>(&backend.inner.stream, spec.elements()? / 2)?;
        let mut packed_scales = backend
            .inner
            .pool
            .allocate::<u8>(&backend.inner.stream, spec.scale_elements()?)?;
        let mut weight_offset = 0;
        let mut scale_offset = 0;
        for source in weights {
            let weight_end = source.weight.len();
            backend.inner.stream.copy_device_range(
                &source.weight,
                0..weight_end,
                &mut packed_weight,
                weight_offset,
            )?;
            let scale_end = source.weight_scales.len();
            backend.inner.stream.copy_device_range(
                &source.weight_scales,
                0..scale_end,
                &mut packed_scales,
                scale_offset,
            )?;
            weight_offset += weight_end;
            scale_offset += scale_end;
        }
        Ok(Self {
            weight: packed_weight,
            weight_scales: packed_scales,
            input_scale: first.input_scale.clone(),
            input_global: first.input_global,
            alpha: first.alpha,
            config: NvFp4Config::new(input_features, output_features),
        })
    }
}

fn read_scalar(backend: &CudaBackend, source: &DeviceBuffer<f32>) -> Result<f32> {
    let mut host = backend.inner.context.allocate_pinned::<f32>(1)?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    let values = host.to_vec()?;
    values.first().copied().ok_or_else(|| Error::InvalidTensorSize {
        name: "NVFP4 global scale".into(),
        expected: 1,
        actual: values.len(),
    })
}

fn copy_scalar(backend: &CudaBackend, value: f32) -> Result<DeviceBuffer<f32>> {
    let mut host = backend.inner.context.allocate_pinned::<f32>(1)?;
    host.copy_from_slice(&[value])?;
    let mut device = backend.inner.pool.allocate::<f32>(&backend.inner.stream, 1)?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.inner.stream.synchronize()?;
    Ok(device)
}
