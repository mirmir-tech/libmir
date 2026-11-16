use std::path::Path;

use super::{
    GgmlType,
    reader::{Parsed, Reader, invalid},
};
use crate::{error::Result, weights::TensorInfo};

/// GGML tensors have at most four dimensions (`GGML_MAX_DIMS`).
const MAX_DIMS: u32 = 4;

/// One entry of the GGUF tensor directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    /// GGML order: `dims[0]` is the contiguous (innermost) dimension.
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset from the start of the aligned data section.
    pub offset: u64,
    /// Payload bytes, derived from the type's block geometry.
    pub bytes: u64,
}

impl GgufTensor {
    pub(super) fn read(reader: &mut Reader<'_>) -> Parsed<Self> {
        let name = reader.string()?;
        let rank = reader.u32()?;
        if rank == 0 || rank > MAX_DIMS {
            return Err(invalid(format!("tensor `{name}` has {rank} dimensions")));
        }
        let mut dims = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            let dim = reader.u64()?;
            if dim == 0 || i64::try_from(dim).is_err() {
                return Err(invalid(format!("tensor `{name}` has dimension {dim}")));
            }
            dims.push(dim);
        }
        let ggml_type = GgmlType::from_id(reader.u32()?)?;
        let offset = reader.u64()?;
        let bytes = payload_bytes(&name, &dims, ggml_type)?;
        Ok(Self { name, dims, ggml_type, offset, bytes })
    }

    #[must_use]
    pub fn elements(&self) -> u64 {
        // Checked when the tensor was read.
        self.dims.iter().product()
    }

    /// Row-major shape as `SafeTensors` writes it: outermost dimension first.
    pub fn shape(&self) -> Result<Vec<usize>> {
        self.dims.iter().rev().map(|&dim| Ok(usize::try_from(dim)?)).collect()
    }

    #[must_use]
    pub fn end(&self) -> u64 {
        // Checked against the data section when the header was validated.
        self.offset.saturating_add(self.bytes)
    }

    /// The catalog entry the binding layer reads; `dtype` is the GGML name.
    pub fn info(&self, file: &Path, data_start: u64) -> Result<TensorInfo> {
        Ok(TensorInfo {
            name: self.name.clone(),
            file: file.to_path_buf(),
            dtype: self.ggml_type.name().to_owned(),
            shape: self.shape()?,
            data_start,
            data_offsets: [self.offset, self.end()],
        })
    }
}

fn payload_bytes(name: &str, dims: &[u64], ggml_type: GgmlType) -> Parsed<u64> {
    let overflow = || invalid(format!("tensor `{name}` is too large"));
    let row = ggml_type.row_bytes(dims[0]).map_err(|_| {
        invalid(format!(
            "tensor `{name}` of type {ggml_type} has rows of {} values, not whole blocks of {}",
            dims[0],
            ggml_type.block_elements()
        ))
    })?;
    let mut bytes = row;
    let mut elements = dims[0];
    for &dim in &dims[1..] {
        bytes = bytes.checked_mul(dim).ok_or_else(overflow)?;
        elements = elements.checked_mul(dim).ok_or_else(overflow)?;
    }
    if i64::try_from(elements).is_err() {
        return Err(overflow());
    }
    Ok(bytes)
}
