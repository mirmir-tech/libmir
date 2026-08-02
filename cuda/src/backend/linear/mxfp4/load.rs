use mircuda::DeviceBuffer;
use models::weights::{BlockQuantization, BlockStorageDType, TensorBinding, TensorStorage};

use super::{CudaTensorSet, MxFp4CheckpointWeight, projection_shape, unsupported};
use crate::{CudaTensorDType, Error, Result};

impl MxFp4CheckpointWeight {
    pub fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        let TensorStorage::BlockQuantized {
            format,
            scales,
            global_scale: None,
            input_scale: None,
            bias,
            packing: _,
        } = &binding.storage
        else {
            return Err(unsupported(binding, "requires self-contained MXFP4 storage"));
        };
        if !format.is_mxfp4() {
            return Err(unsupported(binding, "does not match the MXFP4 contract"));
        }
        let (layout, prefix, output_features, input_features) = projection_shape(binding)?;
        let packed = packed(tensors, binding, *format, &prefix, output_features, input_features)?;
        let mut scale_shape = prefix.clone();
        scale_shape.extend([output_features, input_features / 32]);
        let scales = tensor(tensors, scales, CudaTensorDType::U8, "U8")?;
        require_shape(&scales, &scale_shape)?;
        let mut bias_shape = prefix;
        bias_shape.push(output_features);
        let bias = bias
            .as_deref()
            .map(|name| tensor(tensors, name, CudaTensorDType::Bf16, "BF16"))
            .transpose()?;
        if let Some(bias) = &bias {
            require_shape(bias, &bias_shape)?;
        }
        Ok(Self {
            packed,
            scales,
            bias,
            input_features,
            output_features,
            layout,
        })
    }
}

fn packed(
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
    format: BlockQuantization,
    prefix: &[usize],
    output: usize,
    input: usize,
) -> Result<DeviceBuffer<u8>> {
    let (dtype, expected, tail) = match format.storage_dtype {
        BlockStorageDType::U8 => (CudaTensorDType::U8, "U8", vec![input / 32, 16]),
        BlockStorageDType::U32 => (CudaTensorDType::U32, "U32", vec![input / 8]),
        _ => return Err(unsupported(binding, "uses an unsupported MXFP4 container dtype")),
    };
    let mut shape = prefix.to_vec();
    shape.push(output);
    shape.extend(tail);
    let weight = tensor(tensors, &binding.source, dtype, expected)?;
    require_shape(&weight, &shape)?;
    let packed = match format.storage_dtype {
        BlockStorageDType::U8 => weight.as_u8().cloned(),
        BlockStorageDType::U32 => weight.as_u32().map(DeviceBuffer::reinterpret).transpose()?,
        _ => None,
    }
    .ok_or_else(|| Error::DTypeMismatch { name: binding.source.clone(), expected })?;
    Ok(packed)
}

fn tensor(
    tensors: &CudaTensorSet,
    name: &str,
    dtype: CudaTensorDType,
    expected: &'static str,
) -> Result<crate::CudaTensor> {
    let tensor = tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
    if tensor.dtype() != dtype {
        return Err(Error::DTypeMismatch { name: name.into(), expected });
    }
    Ok(tensor.clone())
}

fn require_shape(tensor: &crate::CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.into(),
            actual: tensor.shape().into(),
        })
    }
}
