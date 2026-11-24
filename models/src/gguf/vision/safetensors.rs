//! A minimal `SafeTensors` writer for the few tensors an import assembles.

use serde_json::{Map, Value, json};

use crate::error::Result;

/// One file of `(name, dtype, shape, bytes)` tensors.
pub(super) fn file(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut header = Map::new();
    let mut offset = 0_usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}),
        );
        offset += bytes.len();
    }
    let mut text = serde_json::to_vec(&Value::Object(header))?;
    // The data section starts 8-byte aligned.
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let mut file = u64::try_from(text.len())?.to_le_bytes().to_vec();
    file.extend(text);
    for (_, _, _, bytes) in tensors {
        file.extend(bytes);
    }
    Ok(file)
}
