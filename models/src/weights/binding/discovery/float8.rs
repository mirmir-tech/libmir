use std::{collections::BTreeSet, fs};

use serde_json::Value;

use crate::{
    error::{ModelsError, Result},
    weights::{
        Float8ActivationScale, Float8Format, Float8ParameterDType, Float8Quantization,
        Float8ScaleGranularity, Float8ScaleMode, TensorCatalog, TensorInfo, TensorStorage,
    },
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Float8Hint {
    activation_scale: Float8ActivationScale,
    block_shape: Option<[usize; 2]>,
}

pub(super) fn hint(layout: &crate::layout::ModelLayout) -> Result<Option<Float8Hint>> {
    let source = fs::read_to_string(&layout.config_path)?;
    let root: Value = serde_json::from_str(&source)?;
    let Some(config) = ["quantization_config", "quantization"]
        .into_iter()
        .find_map(|key| root.get(key))
    else {
        return Ok(None);
    };
    if !config
        .get("quant_method")
        .and_then(Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case("compressed-tensors"))
    {
        return Ok(None);
    }
    let Some(groups) = config.get("config_groups").and_then(Value::as_object) else {
        return Ok(None);
    };
    let mut activation_scale = None;
    let mut block_shape = None;
    for group in groups.values() {
        let Some(weights) = group.get("weights") else {
            continue;
        };
        if weights.get("type").and_then(Value::as_str) != Some("float")
            || weights.get("num_bits").and_then(Value::as_u64) != Some(8)
        {
            continue;
        }
        let input = group.get("input_activations");
        let mode = if input.and_then(|value| value.get("dynamic")).and_then(Value::as_bool)
            == Some(true)
            && input.and_then(|value| value.get("strategy")).and_then(Value::as_str)
                == Some("token")
        {
            Float8ActivationScale::DynamicToken
        } else {
            Float8ActivationScale::None
        };
        if activation_scale.replace(mode).is_some_and(|previous| previous != mode) {
            return Err(invalid("quantization_config", "mixed FP8 activation contracts"));
        }
        let shape = block_shape_from(weights)?;
        if let Some(shape) = shape
            && block_shape.replace(shape).is_some_and(|previous| previous != shape)
        {
            return Err(invalid("quantization_config", "mixed FP8 block dimensions"));
        }
    }
    Ok(activation_scale.map(|activation_scale| Float8Hint { activation_scale, block_shape }))
}

fn block_shape_from(weights: &Value) -> Result<Option<[usize; 2]>> {
    if weights.get("strategy").and_then(Value::as_str) != Some("block") {
        return Ok(None);
    }
    let values = weights.get("block_structure").and_then(Value::as_array).ok_or_else(|| {
        invalid("quantization_config", "block FP8 weights have no block_structure")
    })?;
    if values.len() != 2 {
        return Err(invalid("quantization_config", "FP8 block_structure must have two dimensions"));
    }
    let output = usize::try_from(values[0].as_u64().unwrap_or(0)).unwrap_or(0);
    let input = usize::try_from(values[1].as_u64().unwrap_or(0)).unwrap_or(0);
    if output == 0 || input == 0 {
        return Err(invalid("quantization_config", "FP8 block dimensions must be positive"));
    }
    Ok(Some([output, input]))
}

pub(super) fn storage(
    prefix: &str,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
    hint: Option<Float8Hint>,
) -> Result<Option<TensorStorage>> {
    let Some(value_format) = Float8Format::parse(tensor)? else {
        return Ok(None);
    };
    let multiplier = format!("{prefix}.weight_scale");
    let inverse = format!("{prefix}.weight_scale_inv");
    if catalog.contains(&multiplier) && catalog.contains(&inverse) {
        return Err(invalid(prefix, "both multiplier and inverse scales are present"));
    }
    let (scale_mode, scale) = if catalog.contains(&multiplier) {
        (Float8ScaleMode::Multiplier, Some(multiplier))
    } else if catalog.contains(&inverse) {
        (Float8ScaleMode::InverseMultiplier, Some(inverse))
    } else {
        (Float8ScaleMode::None, None)
    };
    let input_scale = catalog
        .contains(&format!("{prefix}.input_scale"))
        .then(|| format!("{prefix}.input_scale"));
    let bias = catalog.contains(&format!("{prefix}.bias")).then(|| format!("{prefix}.bias"));
    let scale_granularity = scale
        .as_deref()
        .map(|name| granularity(tensor, catalog, name, hint.and_then(|value| value.block_shape)))
        .transpose()?
        .unwrap_or(Float8ScaleGranularity::None);
    let scale_dtype = scale.as_deref().map(|name| parameter_dtype(catalog, name)).transpose()?;
    let input_scale_dtype =
        input_scale.as_deref().map(|name| parameter_dtype(catalog, name)).transpose()?;
    let activation_scale = if input_scale.is_some() {
        Float8ActivationScale::StaticTensor
    } else {
        hint.map_or(Float8ActivationScale::None, |value| value.activation_scale)
    };
    for name in scale.iter().chain(&input_scale).chain(&bias) {
        let _inserted = consumed.insert(name.clone());
    }
    Ok(Some(TensorStorage::Float8 {
        format: Float8Quantization {
            format: value_format,
            scale_mode,
            scale_granularity,
            scale_dtype,
            activation_scale,
            input_scale_dtype,
        },
        scale,
        input_scale,
        bias,
    }))
}

fn parameter_dtype(catalog: &TensorCatalog, name: &str) -> Result<Float8ParameterDType> {
    Float8ParameterDType::parse(
        catalog.get(name).ok_or_else(|| invalid(name, "parameter tensor is missing"))?,
    )
}

fn granularity(
    weight: &TensorInfo,
    catalog: &TensorCatalog,
    scale: &str,
    block_shape: Option<[usize; 2]>,
) -> Result<Float8ScaleGranularity> {
    let shape = &catalog
        .get(scale)
        .ok_or_else(|| invalid(scale, "scale tensor is missing"))?
        .shape;
    if shape.is_empty() || shape == &[1] {
        return Ok(Float8ScaleGranularity::Tensor);
    }
    let Some((input, output)) = weight.shape.split_last() else {
        return Err(invalid(&weight.name, "weight shape is empty"));
    };
    if let Some([output_block_size, input_block_size]) = block_shape {
        let output_size = output
            .last()
            .copied()
            .ok_or_else(|| invalid(&weight.name, "block-grid weight is not a matrix"))?;
        let expected = [output_size.div_ceil(output_block_size), input.div_ceil(input_block_size)];
        if weight.shape.len() != shape.len()
            || weight.shape[..weight.shape.len() - 2] != shape[..shape.len() - 2]
            || shape[shape.len() - 2..] != expected
        {
            return Err(invalid(scale, "scale grid does not match declared FP8 block dimensions"));
        }
        return Ok(Float8ScaleGranularity::BlockGrid {
            output_groups: expected[0],
            input_groups: expected[1],
            output_block_size: Some(output_block_size),
            input_block_size: Some(input_block_size),
        });
    }
    if shape == output
        || (shape.len() == weight.shape.len()
            && shape.last() == Some(&1)
            && shape[..shape.len() - 1] == *output)
    {
        return Ok(Float8ScaleGranularity::OutputChannel);
    }
    if weight.shape.len() == shape.len()
        && weight.shape[..weight.shape.len() - 2] == shape[..shape.len() - 2]
    {
        let output_groups = shape[shape.len() - 2];
        let input_groups = shape[shape.len() - 1];
        let output_size = output[output.len() - 1];
        if output_groups > 0
            && input_groups > 0
            && output_groups <= output_size
            && input_groups <= *input
        {
            return Ok(Float8ScaleGranularity::BlockGrid {
                output_groups,
                input_groups,
                output_block_size: None,
                input_block_size: None,
            });
        }
    }
    Err(invalid(
        scale,
        "scale shape is not tensor, output-channel, or block-grid geometry",
    ))
}

fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid float8 binding {name}: {reason}"))
}

#[cfg(test)]
#[path = "float8/tests.rs"]
mod tests;
