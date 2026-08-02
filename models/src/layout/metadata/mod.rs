use std::{fs, path::Path};

use foundation::model::Quantization;
use serde_json::Value;

use crate::{error::Result, layout::ModelLayout};

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub dtype: Option<String>,
    pub context_len: usize,
    pub quantization: Quantization,
    pub quantization_group_size: Option<usize>,
    pub quantization_mode: Option<String>,
}

impl ModelMetadata {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let mut metadata = Self::from_config_path(&layout.config_path)?;
        if let Some(path) = layout.configuration_path.as_deref() {
            let json = fs::read_to_string(path)?;
            let value: Value = serde_json::from_str(&json)?;
            if let Some(context) = read_context_len(&value)? {
                metadata.context_len = metadata.context_len.max(context);
            }
        }
        Ok(metadata)
    }

    pub fn from_config_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&json)?;
        let architectures = read_architectures(&value);
        let model_type = value.get("model_type").and_then(Value::as_str).map(str::to_owned);
        let dtype = read_dtype(&value);
        let context_len = read_context_len(&value)?.unwrap_or(4096);
        let quantization = read_quantization(&value);
        let quantization_group_size = read_quantization_usize(&value, "group_size")?;
        let quantization_mode = read_quantization_string(&value, "mode");
        Ok(Self {
            architectures,
            model_type,
            dtype,
            context_len,
            quantization,
            quantization_group_size,
            quantization_mode,
        })
    }
}

fn read_architectures(value: &Value) -> Vec<String> {
    value
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn read_context_len(value: &Value) -> Result<Option<usize>> {
    for section in config_sections(value) {
        for key in ["max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"] {
            if let Some(raw) = section.get(key).and_then(Value::as_u64) {
                return Ok(Some(usize::try_from(raw)?));
            }
        }
    }
    Ok(None)
}

fn config_sections(value: &Value) -> Vec<&Value> {
    let mut sections: Vec<&Value> = ["text_config", "language_config"]
        .into_iter()
        .filter_map(|key| value.get(key))
        .filter(|section| section.is_object())
        .collect();
    sections.push(value);
    sections
}

fn read_quantization(value: &Value) -> Quantization {
    for key in ["quantization_config", "quantization"] {
        if let Some(quantization) = config_quantization(value.get(key)) {
            return quantization;
        }
    }
    dtype_quantization(value)
}

fn config_quantization(config: Option<&Value>) -> Option<Quantization> {
    let config = config?;
    if let Some(name) = quantization_name(config) {
        if name.eq_ignore_ascii_case("bitsandbytes") {
            return match config.get("bnb_4bit_quant_type").and_then(Value::as_str) {
                Some(kind) if kind.eq_ignore_ascii_case("nf4") => Some(Quantization::Nf4),
                Some(kind) if kind.eq_ignore_ascii_case("fp4") => Some(Quantization::Fp4),
                _ => Some(Quantization::Custom("bitsandbytes".into())),
            };
        }
        if name.eq_ignore_ascii_case("nvfp4") {
            return Some(Quantization::NvFp4);
        }
        if name.eq_ignore_ascii_case("mxfp4") {
            return Some(Quantization::MxFp4);
        }
        if name.eq_ignore_ascii_case("mxfp8") {
            return Some(Quantization::MxFp8);
        }
    }
    let bits = config
        .get("bits")
        .or_else(|| {
            config
                .get("config_groups")?
                .as_object()?
                .values()
                .find_map(|group| group.pointer("/weights/num_bits"))
        })
        .and_then(Value::as_u64);
    match bits {
        Some(4) => Some(Quantization::Int4),
        Some(8) => Some(Quantization::Int8),
        Some(other) => Some(Quantization::Custom(format!("{other}bit"))),
        None => None,
    }
}

fn quantization_name(config: &Value) -> Option<&str> {
    ["quant_algo", "quant_method", "mode"]
        .into_iter()
        .find_map(|field| config.get(field).and_then(Value::as_str))
}

fn dtype_quantization(value: &Value) -> Quantization {
    read_dtype(value).map_or(Quantization::None, |dtype| dtype_name_quantization(&dtype))
}

fn dtype_name_quantization(dtype: &str) -> Quantization {
    match dtype {
        "float16" => Quantization::F16,
        "bfloat16" => Quantization::Bf16,
        other => Quantization::Custom(other.to_owned()),
    }
}

fn read_dtype(value: &Value) -> Option<String> {
    for section in config_sections(value) {
        for key in ["torch_dtype", "dtype"] {
            if let Some(dtype) = section.get(key).and_then(Value::as_str) {
                return Some(dtype.to_owned());
            }
        }
    }
    None
}

fn read_quantization_usize(value: &Value, field: &str) -> Result<Option<usize>> {
    for key in ["quantization_config", "quantization"] {
        let Some(section) = value.get(key) else {
            continue;
        };
        if let Some(number) = section.get(field).and_then(Value::as_u64) {
            return Ok(Some(usize::try_from(number)?));
        }
        if let Some(number) = section
            .pointer(&format!("/config_groups/group_0/weights/{field}"))
            .and_then(Value::as_u64)
        {
            return Ok(Some(usize::try_from(number)?));
        }
    }
    Ok(None)
}

fn read_quantization_string(value: &Value, field: &str) -> Option<String> {
    for key in ["quantization_config", "quantization"] {
        if let Some(text) =
            value.get(key).and_then(|section| section.get(field)).and_then(Value::as_str)
        {
            return Some(text.to_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests;
