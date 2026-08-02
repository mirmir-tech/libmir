use super::{companion, companion_dtype, dense_shape, invalid, shape};
use crate::{
    error::Result,
    weights::{
        Float8ActivationScale, Float8ScaleGranularity, Float8ScaleMode, TensorBinding,
        TensorCatalog, TensorStorage,
    },
};

pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    catalog: &TensorCatalog,
) -> Result<()> {
    let TensorStorage::Float8 { format, scale, input_scale, bias } = &binding.storage else {
        return Err(invalid(&binding.source, "storage is not float8"));
    };
    shape(&binding.source, &binding.shape, &dense_shape(binding, logical))?;
    companion_dtype(catalog, &binding.source, format.format.as_str())?;
    let scale_required = format.scale_mode != Float8ScaleMode::None;
    if scale_required != scale.is_some()
        || (format.scale_granularity == Float8ScaleGranularity::None) != scale.is_none()
        || format.scale_dtype.is_some() != scale.is_some()
        || format.input_scale_dtype.is_some() != input_scale.is_some()
        || matches!(format.activation_scale, Float8ActivationScale::StaticTensor)
            != input_scale.is_some()
    {
        return Err(invalid(&binding.source, "scale hierarchy differs from float8 contract"));
    }
    if let Some(scale) = scale {
        let expected = scale_shape(binding, format.scale_granularity)?;
        if format.scale_granularity == Float8ScaleGranularity::Tensor {
            let actual = &catalog
                .get(scale)
                .ok_or_else(|| invalid(scale, "scale tensor is missing"))?
                .shape;
            if !actual.is_empty() && actual != &[1] {
                return Err(invalid(scale, "tensor scale is not scalar"));
            }
        } else if format.scale_granularity == Float8ScaleGranularity::OutputChannel {
            let actual = &catalog
                .get(scale)
                .ok_or_else(|| invalid(scale, "scale tensor is missing"))?
                .shape;
            let mut singleton = expected.clone();
            singleton.push(1);
            if actual != &expected && actual != &singleton {
                return Err(invalid(scale, "output-channel scale shape does not match weight"));
            }
        } else {
            companion(catalog, scale, &expected)?;
        }
        companion_dtype(
            catalog,
            scale,
            format
                .scale_dtype
                .ok_or_else(|| invalid(scale, "scale dtype is missing"))?
                .as_str(),
        )?;
    }
    if let Some(input_scale) = input_scale {
        let actual = &catalog
            .get(input_scale)
            .ok_or_else(|| invalid(input_scale, "input scale tensor is missing"))?
            .shape;
        if !actual.is_empty() && actual != &[1] {
            return Err(invalid(input_scale, "input scale is not scalar"));
        }
        companion_dtype(
            catalog,
            input_scale,
            format
                .input_scale_dtype
                .ok_or_else(|| invalid(input_scale, "input scale dtype is missing"))?
                .as_str(),
        )?;
    }
    if let Some(bias) = bias
        && logical.len() > 1
    {
        companion(catalog, bias, &logical[..logical.len() - 1])?;
    }
    Ok(())
}

fn scale_shape(binding: &TensorBinding, granularity: Float8ScaleGranularity) -> Result<Vec<usize>> {
    match granularity {
        Float8ScaleGranularity::None => Err(invalid(&binding.source, "scale geometry is missing")),
        Float8ScaleGranularity::Tensor => Ok(Vec::new()),
        Float8ScaleGranularity::OutputChannel => binding
            .shape
            .split_last()
            .map(|(_input, output)| output.to_vec())
            .ok_or_else(|| invalid(&binding.source, "weight shape is empty")),
        Float8ScaleGranularity::BlockGrid {
            output_groups,
            input_groups,
            output_block_size,
            input_block_size,
        } => {
            if binding.shape.len() < 2 {
                return Err(invalid(&binding.source, "block-grid weight is not a matrix"));
            }
            if output_block_size.is_some() != input_block_size.is_some() {
                return Err(invalid(&binding.source, "FP8 block dimensions are incomplete"));
            }
            if let (Some(output_block), Some(input_block)) = (output_block_size, input_block_size) {
                let input = binding.shape[binding.shape.len() - 1];
                let output = binding.shape[binding.shape.len() - 2];
                if output_block == 0
                    || input_block == 0
                    || output_groups != output.div_ceil(output_block)
                    || input_groups != input.div_ceil(input_block)
                {
                    return Err(invalid(
                        &binding.source,
                        "FP8 scale grid differs from declared block dimensions",
                    ));
                }
            }
            let mut shape = binding.shape[..binding.shape.len() - 2].to_vec();
            shape.extend([output_groups, input_groups]);
            Ok(shape)
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::weights::{
        Float8Format, Float8Quantization, LogicalTensorRole, TensorInfo, TensorStorage,
    };

    #[test]
    fn rejects_scale_hierarchy_divergence() {
        let binding = TensorBinding {
            role: LogicalTensorRole::Output,
            source: "projection.weight".into(),
            shape: vec![4, 8],
            logical_shape: Some(vec![4, 8]),
            transforms: Vec::new(),
            storage: TensorStorage::Float8 {
                format: Float8Quantization {
                    format: Float8Format::E4M3,
                    scale_mode: Float8ScaleMode::Multiplier,
                    scale_granularity: Float8ScaleGranularity::None,
                    scale_dtype: None,
                    activation_scale: Float8ActivationScale::None,
                    input_scale_dtype: None,
                },
                scale: None,
                input_scale: None,
                bias: None,
            },
        };
        let catalog = TensorCatalog {
            tensors: vec![TensorInfo {
                name: binding.source.clone(),
                file: PathBuf::new(),
                dtype: "F8_E4M3".into(),
                shape: binding.shape.clone(),
                data_start: 0,
                data_offsets: [0, 0],
            }],
        };
        assert!(validate(&binding, &[4, 8], &catalog).is_err());
    }
}
