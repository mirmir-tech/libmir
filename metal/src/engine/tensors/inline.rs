use std::{collections::BTreeMap, sync::Mutex};

use models::weights::{DenseDType, TensorCatalog, TensorDType};

use super::TensorFile;
use crate::engine::{Array, Dtype, Error, Result, Stream};

impl TensorFile {
    /// Parses an in-memory `SafeTensors` buffer (such as an assembled GGUF
    /// vision projection) directly into arrays without writing a file to disk.
    pub fn from_safetensors_bytes(name: &str, bytes: &[u8], stream: &Stream) -> Result<Self> {
        let catalog = TensorCatalog::from_safetensors_header(name, bytes)?;
        let mut converted = BTreeMap::new();
        for tensor in &catalog.tensors {
            let start = usize::try_from(tensor.payload_start()?)?;
            let length = tensor.payload_bytes()?;
            let payload = bytes.get(start..start + length).ok_or_else(|| {
                Error::InvalidModel(format!("inline tensor `{}` out of bounds", tensor.name))
            })?;
            let shape = tensor
                .shape
                .iter()
                .map(|&dimension| Ok(i32::try_from(dimension)?))
                .collect::<Result<Vec<_>>>()?;
            let (values, target) = match &tensor.dtype {
                TensorDType::Dense(DenseDType::F32) => (decode_f32(payload), Dtype::Float32),
                TensorDType::Dense(DenseDType::F16) => (decode_f16(payload), Dtype::Float16),
                TensorDType::Dense(DenseDType::Bf16) => (decode_bf16(payload), Dtype::Bfloat16),
                other => {
                    return Err(Error::InvalidModel(format!(
                        "inline tensor `{}` has unsupported dtype {other}",
                        tensor.name
                    )));
                },
            };
            let array = Array::from_f32(&values, &shape)?;
            let array = if target == Dtype::Float32 {
                array
            } else {
                array.astype(target, stream)?
            };
            let _prev = converted.insert(tensor.name.clone(), array.native().clone());
        }
        Ok(Self {
            native: None,
            converted: Mutex::new(converted),
        })
    }
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(chunk)) << 16))
        .collect()
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(|chunk| f16_to_f32(u16::from_le_bytes(chunk)))
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127_u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 31 {
        sign | (0xff << 23) | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}
