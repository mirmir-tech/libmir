use models::weights::{Float8ParameterDType, TensorBinding, TensorStorage};

use super::{
    CudaTensorDType, CudaTensorSet, DirectFp8CheckpointWeight, DirectFp8Format, Result,
    execution_contract, tensor, unsupported,
};

impl DirectFp8CheckpointWeight {
    pub fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        let TensorStorage::Float8 { format, scale, input_scale, bias } = &binding.storage else {
            return Err(unsupported(&binding.source, "requires direct FP8 storage"));
        };
        if !binding.transforms.is_empty() {
            return Err(unsupported(&binding.source, "does not support storage transforms"));
        }
        let [output_features, input_features] = binding
            .logical_shape
            .as_deref()
            .ok_or_else(|| unsupported(&binding.source, "requires a logical matrix shape"))?
        else {
            return Err(unsupported(&binding.source, "requires a logical matrix shape"));
        };
        let (value_format, scale_geometry, inverse_scale, activation) =
            execution_contract(&binding.source, *format, *input_features, *output_features)?;
        let (weight_dtype, weight_name) = match value_format {
            DirectFp8Format::E4M3 => (CudaTensorDType::F8E4M3, "F8_E4M3"),
            DirectFp8Format::E5M2 => (CudaTensorDType::F8E5M2, "F8_E5M2"),
        };
        let weight = tensor(tensors, &binding.source, weight_dtype, weight_name)?;
        let scales = scale_tensor(tensors, &binding.source, scale.as_deref(), format.scale_dtype)?;
        let input_scale = scale_tensor(
            tensors,
            &binding.source,
            input_scale.as_deref(),
            format.input_scale_dtype,
        )?;
        let bias = bias
            .as_deref()
            .map(|name| tensor(tensors, name, CudaTensorDType::Bf16, "BF16"))
            .transpose()?;
        Ok(Self {
            weight,
            scales,
            input_scale,
            bias,
            input_features: *input_features,
            output_features: *output_features,
            format: value_format,
            scale: scale_geometry,
            inverse_scale,
            activation,
        })
    }
}

fn scale_tensor(
    tensors: &CudaTensorSet,
    source: &str,
    name: Option<&str>,
    dtype: Option<Float8ParameterDType>,
) -> Result<Option<super::CudaTensor>> {
    match (name, dtype) {
        (Some(name), Some(Float8ParameterDType::BF16)) => {
            Ok(Some(tensor(tensors, name, CudaTensorDType::Bf16, "BF16")?))
        },
        (Some(name), Some(Float8ParameterDType::F32)) => {
            Ok(Some(tensor(tensors, name, CudaTensorDType::F32, "F32")?))
        },
        (None, None) => Ok(None),
        _ => Err(unsupported(source, "has an inconsistent scale tensor")),
    }
}
