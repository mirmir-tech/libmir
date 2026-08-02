use std::{collections::BTreeSet, fs};

use serde_json::Value;

use super::super::{
    GptqBits, GptqCheckpointFormat, GptqPacking, GptqQuantization, GptqScaleDType,
    GptqStorageDType, TensorStorage,
};
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

#[derive(Debug, Clone, Copy)]
pub(super) struct GptqHint {
    bits: GptqBits,
    group_size: Option<usize>,
    checkpoint_format: GptqCheckpointFormat,
    symmetric: bool,
    activation_order: bool,
}

pub(super) fn hint(layout: &ModelLayout) -> Result<Option<GptqHint>> {
    let source = fs::read_to_string(&layout.config_path)?;
    let root: Value = serde_json::from_str(&source)?;
    let Some(config) = ["quantization_config", "quantization"]
        .into_iter()
        .find_map(|key| root.get(key))
    else {
        return Ok(None);
    };
    let method = config
        .get("method")
        .or_else(|| config.get("quant_method"))
        .and_then(Value::as_str);
    if !method.is_some_and(|method| method.eq_ignore_ascii_case("gptq")) {
        return Ok(None);
    }
    let bits = required_u8(config, "bits").and_then(GptqBits::try_from)?;
    let group_size = match config.get("group_size").and_then(Value::as_i64) {
        Some(-1) => None,
        Some(value) if value > 0 => Some(usize::try_from(value)?),
        _ => return Err(invalid("group_size must be positive or -1")),
    };
    let checkpoint_format = match config
        .get("checkpoint_format")
        .or_else(|| config.get("format"))
        .and_then(Value::as_str)
        .unwrap_or("gptq")
        .to_ascii_lowercase()
        .as_str()
    {
        "gptq" => GptqCheckpointFormat::Gptq,
        "gptq_v2" => GptqCheckpointFormat::GptqV2,
        value => return Err(invalid(&format!("unsupported checkpoint format {value}"))),
    };
    let pack = config.get("pack_dtype").and_then(Value::as_str).unwrap_or("int32");
    if !pack.eq_ignore_ascii_case("int32") {
        return Err(invalid("only int32 packing is typed"));
    }
    Ok(Some(GptqHint {
        bits,
        group_size,
        checkpoint_format,
        symmetric: required_bool(config, "sym")?,
        activation_order: required_bool(config, "desc_act")?,
    }))
}

pub(super) fn storage(
    prefix: &str,
    logical: Option<&[usize]>,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
    hint: GptqHint,
) -> Result<TensorStorage> {
    let [output, input] = logical.ok_or_else(|| invalid("logical shape is unavailable"))? else {
        return Err(invalid("logical tensor is not a matrix"));
    };
    let group_size = hint.group_size.unwrap_or(*input);
    let groups = divided(*input, group_size, &tensor.name)?;
    let bits = usize::from(hint.bits.get());
    let packed_input = divided(
        input.checked_mul(bits).ok_or_else(|| invalid("width overflow"))?,
        32,
        &tensor.name,
    )?;
    let packed_output = divided(
        output.checked_mul(bits).ok_or_else(|| invalid("width overflow"))?,
        32,
        &tensor.name,
    )?;
    require(tensor, "I32", &[packed_input, *output])?;
    let scales = companion(prefix, "scales");
    let zero_points = companion(prefix, "qzeros");
    let group_indices = companion(prefix, "g_idx");
    let scale = required(catalog, &scales)?;
    require(scale, scale.dtype.as_str(), &[groups, *output])?;
    require(required(catalog, &zero_points)?, "I32", &[groups, packed_output])?;
    require(required(catalog, &group_indices)?, "I32", &[*input])?;
    consumed.extend([scales.clone(), zero_points.clone(), group_indices.clone()]);
    Ok(TensorStorage::Gptq {
        format: GptqQuantization {
            bits: hint.bits,
            group_size,
            packing: GptqPacking::InputLittleEndian,
            storage_dtype: GptqStorageDType::I32,
            scale_dtype: GptqScaleDType::parse(scale)?,
            checkpoint_format: hint.checkpoint_format,
            symmetric: hint.symmetric,
            activation_order: hint.activation_order,
            packed_zero_points: true,
        },
        scales,
        zero_points,
        group_indices,
    })
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog.get(name).ok_or_else(|| invalid(&format!("missing companion {name}")))
}

fn require(tensor: &TensorInfo, dtype: &str, shape: &[usize]) -> Result<()> {
    if tensor.dtype == dtype && tensor.shape == shape {
        Ok(())
    } else {
        Err(invalid(&format!("{} has incompatible dtype or shape", tensor.name)))
    }
}

fn companion(prefix: &str, suffix: &str) -> String {
    format!("{prefix}.{suffix}")
}

fn divided(value: usize, divisor: usize, name: &str) -> Result<usize> {
    value
        .checked_div(divisor)
        .filter(|result| value > 0 && value.is_multiple_of(divisor) && *result > 0)
        .ok_or_else(|| invalid(&format!("{name} is not pack/group aligned")))
}

fn required_u8(config: &Value, field: &str) -> Result<u8> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(&format!("missing {field}")))
        .and_then(|value| u8::try_from(value).map_err(ModelsError::from))
}

fn required_bool(config: &Value, field: &str) -> Result<bool> {
    config
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid(&format!("missing {field}")))
}

fn invalid(reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid GPTQ contract: {reason}"))
}
