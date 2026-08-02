use std::{
    fs,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use serde::Deserialize;
use serde_json::Value;

use super::super::{
    BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
    BitsAndBytesStorageDType, TensorStorage,
};
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
    weights::{TensorCatalog, TensorInfo},
};

#[derive(Clone, Copy)]
pub(super) struct Hint {
    quant_type: BitsAndBytes4BitType,
    compute_dtype: BitsAndBytesComputeDType,
    storage_dtype: BitsAndBytesStorageDType,
    nested: bool,
}

#[derive(Deserialize)]
struct PackedState {
    quant_type: String,
    blocksize: usize,
    dtype: String,
    shape: Vec<usize>,
    nested_blocksize: Option<usize>,
    nested_dtype: Option<String>,
    nested_offset: Option<f32>,
}

pub(super) fn hint(layout: &ModelLayout) -> Result<Option<Hint>> {
    let value: Value = serde_json::from_str(&fs::read_to_string(&layout.config_path)?)?;
    let Some(config) = value.get("quantization_config") else {
        return Ok(None);
    };
    if config.get("quant_method").and_then(Value::as_str) != Some("bitsandbytes") {
        return Ok(None);
    }
    let quant_type = parse_type(required_string(config, "bnb_4bit_quant_type")?)?;
    let compute_dtype =
        BitsAndBytesComputeDType::parse(required_string(config, "bnb_4bit_compute_dtype")?)
            .ok_or_else(|| invalid("unsupported bitsandbytes compute dtype"))?;
    let storage_dtype =
        BitsAndBytesStorageDType::parse(required_string(config, "bnb_4bit_quant_storage")?)
            .ok_or_else(|| invalid("unsupported bitsandbytes storage dtype"))?;
    let nested = config
        .get("bnb_4bit_use_double_quant")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(Some(Hint {
        quant_type,
        compute_dtype,
        storage_dtype,
        nested,
    }))
}

pub(super) fn storage(
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut std::collections::BTreeSet<String>,
    hint: Option<Hint>,
) -> Result<Option<TensorStorage>> {
    let Some(hint) = hint else {
        return Ok(None);
    };
    let absmax = format!("{}.absmax", tensor.name);
    if !catalog.contains(&absmax) {
        return Ok(None);
    }
    let quant_map = format!("{}.quant_map", tensor.name);
    let quant_state =
        format!("{}.quant_state.bitsandbytes__{}", tensor.name, hint.quant_type.as_str());
    let state_tensor = required(catalog, &quant_state)?;
    let state = read_state(state_tensor)?;
    validate_state(tensor, &state, hint)?;
    let nested_absmax = hint.nested.then(|| format!("{}.nested_absmax", tensor.name));
    let nested_quant_map = hint.nested.then(|| format!("{}.nested_quant_map", tensor.name));
    required(catalog, &quant_map)?;
    if let Some(name) = nested_absmax.as_deref() {
        required(catalog, name)?;
    }
    if let Some(name) = nested_quant_map.as_deref() {
        required(catalog, name)?;
    }
    for name in [&absmax, &quant_map, &quant_state] {
        consumed.insert(name.clone());
    }
    consumed.extend(nested_absmax.iter().cloned());
    consumed.extend(nested_quant_map.iter().cloned());
    Ok(Some(TensorStorage::BitsAndBytes4Bit {
        format: BitsAndBytes4BitQuantization {
            quant_type: hint.quant_type,
            block_size: state.blocksize,
            compute_dtype: hint.compute_dtype,
            storage_dtype: hint.storage_dtype,
            nested_block_size: state.nested_blocksize,
        },
        absmax,
        quant_map,
        nested_absmax,
        nested_quant_map,
        quant_state,
        nested_offset_bits: state.nested_offset.map(f32::to_bits),
    }))
}

fn read_state(tensor: &TensorInfo) -> Result<PackedState> {
    if tensor.dtype != "U8" || tensor.payload_bytes()? > 4096 {
        return Err(invalid("bitsandbytes quant_state must be a small U8 tensor"));
    }
    let mut file = File::open(&tensor.file)?;
    file.seek(SeekFrom::Start(tensor.payload_start()?))?;
    let mut bytes = vec![0; tensor.payload_bytes()?];
    file.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn validate_state(tensor: &TensorInfo, state: &PackedState, hint: Hint) -> Result<()> {
    let logical_elements = state.shape.iter().try_fold(1_usize, |n, value| n.checked_mul(*value));
    let payload_bytes = tensor.payload_bytes()?;
    if parse_type(&state.quant_type)? != hint.quant_type
        || logical_elements.is_none_or(|elements| elements.div_ceil(2) != payload_bytes)
        || BitsAndBytesStorageDType::parse(&tensor.dtype) != Some(hint.storage_dtype)
        || (state.nested_blocksize.is_some() != hint.nested)
        || state.nested_offset.is_some() != hint.nested
        || state.nested_dtype.as_deref().is_some_and(|dtype| dtype != "float32")
        || BitsAndBytesComputeDType::parse(&state.dtype).is_none()
    {
        return Err(invalid("bitsandbytes packed state differs from checkpoint config or payload"));
    }
    Ok(())
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .get(name)
        .ok_or_else(|| invalid(&format!("missing bitsandbytes companion {name}")))
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(&format!("missing {key}")))
}

fn parse_type(value: &str) -> Result<BitsAndBytes4BitType> {
    match value.to_ascii_lowercase().as_str() {
        "nf4" => Ok(BitsAndBytes4BitType::Nf4),
        "fp4" => Ok(BitsAndBytes4BitType::Fp4),
        _ => Err(invalid("unsupported bitsandbytes 4-bit type")),
    }
}

fn invalid(detail: &str) -> ModelsError {
    ModelsError::InvalidConfig(detail.into())
}
