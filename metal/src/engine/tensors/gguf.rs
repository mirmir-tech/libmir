//! GGUF tensors as in-place file ranges.
//!
//! Dense GGUF types keep their element type. Block-quantized GGML types stay
//! opaque `Uint8` rows of whole blocks, `[leading..., row_bytes]`, so device
//! memory holds exactly the file's bytes and kernels decode them in place.

use std::path::Path;

use mirtal::{DType, TensorRange};
use models::{
    gguf::{GgmlType, GgufHeader},
    weights::TensorInfo,
};

use crate::engine::{Error, Result};

pub(super) fn ranges(path: &Path) -> Result<Vec<TensorRange>> {
    GgufHeader::read(path)?
        .mapped_catalog(path)?
        .tensors
        .iter()
        .map(range)
        .collect()
}

fn range(tensor: &TensorInfo) -> Result<TensorRange> {
    let ggml =
        GgmlType::from_name(&tensor.dtype).ok_or_else(|| invalid(tensor, "an unknown type"))?;
    let mut shape = tensor
        .shape
        .iter()
        .map(|&dimension| u64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let dtype = match ggml {
        GgmlType::F32 => DType::Float32,
        GgmlType::F16 => DType::Float16,
        GgmlType::BF16 => DType::Bfloat16,
        _ if ggml.is_dense() => return Err(invalid(tensor, "a dense type Metal does not load")),
        _ => {
            let input = shape.pop().ok_or_else(|| invalid(tensor, "no dimensions"))?;
            shape.push(ggml.row_bytes(input)?);
            DType::Uint8
        },
    };
    Ok(TensorRange {
        name: tensor.name.clone(),
        offset: tensor.payload_start()?,
        dtype,
        shape,
    })
}

fn invalid(tensor: &TensorInfo, reason: &str) -> Error {
    Error::InvalidModel(format!("GGUF tensor `{}` has {reason}", tensor.name))
}

/// GGUF stores norms in F32 and some weights in F16; the Metal runtime
/// computes in BF16, as the CUDA loader does for the same files.
pub(super) const fn casts_to_bf16(dtype: DType) -> bool {
    matches!(dtype, DType::Float32 | DType::Float16)
}
