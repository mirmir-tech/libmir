use mircuda::DeviceBuffer;

use super::{
    CudaBackend, CudaTensor, CudaTensorDType, DirectFp8Activation, DirectFp8CheckpointWeight,
    DirectFp8Format, DirectFp8Scale, Error, Result,
};

impl DirectFp8CheckpointWeight {
    pub(in crate::backend) fn pack<const N: usize>(
        backend: &CudaBackend,
        weights: [&Self; N],
    ) -> Result<Option<Self>> {
        let Some(first) = weights.first() else {
            return Ok(None);
        };
        if !weights.iter().all(|weight| compatible(first, weight)) {
            return Ok(None);
        }
        let input_scale = read_scalar(
            backend,
            first
                .input_scale
                .as_ref()
                .ok_or(Error::InvalidExecutionPlan("packed FP8 activation scale is missing"))?,
        )?;
        for weight in weights.iter().skip(1) {
            let scale = weight
                .input_scale
                .as_ref()
                .ok_or(Error::InvalidExecutionPlan("packed FP8 activation scale is missing"))?;
            if read_scalar(backend, scale)?.to_bits() != input_scale.to_bits() {
                return Ok(None);
            }
        }
        let output_features = weights.iter().try_fold(0_usize, |total, weight| {
            total
                .checked_add(weight.output_features)
                .ok_or(Error::InvalidDecoderKernel("packed FP8 output size overflow"))
        })?;
        let elements = output_features
            .checked_mul(first.input_features)
            .ok_or(Error::InvalidDecoderKernel("packed FP8 weight size overflow"))?;
        let mut packed = backend.inner.pool.allocate::<u8>(&backend.inner.stream, elements)?;
        let mut scales = Vec::with_capacity(output_features);
        let mut offset = 0_usize;
        for weight in weights {
            let source = weight.weight.as_f8_e4m3().ok_or_else(|| Error::DTypeMismatch {
                name: weight.weight.name().into(),
                expected: "F8_E4M3",
            })?;
            backend
                .inner
                .stream
                .copy_device_range(source, 0..source.len(), &mut packed, offset)?;
            offset += source.len();
            let scale = read_scalar(
                backend,
                weight
                    .scales
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed FP8 weight scale is missing"))?,
            )?;
            scales.extend(std::iter::repeat_n(scale, weight.output_features));
        }
        let scale_buffer = upload(backend, &scales)?;
        Ok(Some(Self {
            weight: CudaTensor::from_f8_e4m3(
                "packed direct FP8 projection".into(),
                vec![output_features, first.input_features],
                packed,
            ),
            scales: Some(CudaTensor::from_f32(
                "packed direct FP8 projection scales".into(),
                vec![output_features],
                scale_buffer,
            )),
            input_scale: first.input_scale.clone(),
            bias: None,
            input_features: first.input_features,
            output_features,
            format: DirectFp8Format::E4M3,
            scale: DirectFp8Scale::OutputChannel,
            inverse_scale: false,
            activation: DirectFp8Activation::StaticE4M3Tensor,
        }))
    }
}

fn compatible(first: &DirectFp8CheckpointWeight, weight: &DirectFp8CheckpointWeight) -> bool {
    weight.input_features == first.input_features
        && weight.format == DirectFp8Format::E4M3
        && weight.scale == DirectFp8Scale::Tensor
        && !weight.inverse_scale
        && weight.activation == DirectFp8Activation::StaticE4M3Tensor
        && weight.bias.is_none()
        && weight
            .scales
            .as_ref()
            .is_some_and(|scale| scale.dtype() == CudaTensorDType::F32)
        && weight
            .input_scale
            .as_ref()
            .is_some_and(|scale| scale.dtype() == CudaTensorDType::F32)
}

fn read_scalar(backend: &CudaBackend, tensor: &CudaTensor) -> Result<f32> {
    let source = tensor.as_f32().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "F32",
    })?;
    if source.len() != 1 {
        return Err(Error::InvalidExecutionPlan("packed FP8 scale is not scalar"));
    }
    let mut host = backend.inner.context.allocate_pinned(1)?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    host.to_vec()?
        .into_iter()
        .next()
        .ok_or(Error::InvalidExecutionPlan("packed FP8 scale is empty"))
}

fn upload(backend: &CudaBackend, values: &[f32]) -> Result<DeviceBuffer<f32>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
