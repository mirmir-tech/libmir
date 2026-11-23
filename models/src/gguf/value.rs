//! Metadata values of the GGUF key-value section.

use super::reader::{Parsed, Reader, invalid};

/// Arrays nest at most this deep; GGUF writers produce one level.
const MAX_ARRAY_DEPTH: usize = 2;

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    /// An unsigned integer of any width; negative values are refused.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(value) => Some(u64::from(value)),
            Self::U16(value) => Some(u64::from(value)),
            Self::U32(value) => Some(u64::from(value)),
            Self::U64(value) => Some(value),
            Self::I8(value) => u64::try_from(value).ok(),
            Self::I16(value) => u64::try_from(value).ok(),
            Self::I32(value) => u64::try_from(value).ok(),
            Self::I64(value) => u64::try_from(value).ok(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(value) => Some(i64::from(value)),
            Self::I16(value) => Some(i64::from(value)),
            Self::I32(value) => Some(i64::from(value)),
            Self::I64(value) => Some(value),
            _ => self.as_u64().and_then(|value| i64::try_from(value).ok()),
        }
    }

    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(value) => Some(f64::from(value)),
            Self::F64(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    /// A short description for inventories: scalars in full, arrays by length.
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::String(value) if value.chars().count() > 80 => {
                format!(
                    "{}… ({} characters)",
                    value.chars().take(80).collect::<String>(),
                    value.chars().count()
                )
            },
            Self::String(value) => value.clone(),
            Self::Array(values) => format!("[{} values]", values.len()),
            Self::Bool(value) => value.to_string(),
            Self::F32(value) => value.to_string(),
            Self::F64(value) => value.to_string(),
            other => other.as_i64().map_or_else(|| format!("{other:?}"), |value| value.to_string()),
        }
    }
}

pub fn read_value(
    reader: &mut Reader<'_>,
    kind: u32,
    file_len: Option<u64>,
) -> Parsed<MetadataValue> {
    read_nested(reader, kind, file_len, 0)
}

fn read_nested(
    reader: &mut Reader<'_>,
    kind: u32,
    file_len: Option<u64>,
    depth: usize,
) -> Parsed<MetadataValue> {
    Ok(match kind {
        0 => MetadataValue::U8(reader.u8()?),
        1 => MetadataValue::I8(reader.i8()?),
        2 => MetadataValue::U16(reader.u16()?),
        3 => MetadataValue::I16(reader.i16()?),
        4 => MetadataValue::U32(reader.u32()?),
        5 => MetadataValue::I32(reader.i32()?),
        6 => MetadataValue::F32(reader.f32()?),
        7 => MetadataValue::Bool(reader.bool()?),
        8 => MetadataValue::String(reader.string()?),
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(invalid("metadata arrays nest too deeply"));
            }
            let element = reader.u32()?;
            let count = reader.count("array element", min_bytes(element)?, file_len)?;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_nested(reader, element, file_len, depth + 1)?);
            }
            MetadataValue::Array(values)
        },
        10 => MetadataValue::U64(reader.u64()?),
        11 => MetadataValue::I64(reader.i64()?),
        12 => MetadataValue::F64(reader.f64()?),
        other => return Err(invalid(format!("unknown metadata value type {other}"))),
    })
}

/// The fewest bytes one value of a GGUF type can take.
fn min_bytes(kind: u32) -> Parsed<u64> {
    Ok(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        8 | 10..=12 => 8,
        9 => 12,
        other => return Err(invalid(format!("unknown metadata array type {other}"))),
    })
}
