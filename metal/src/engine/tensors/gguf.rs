//! GGUF tensors as file ranges, each read once into memory on evaluation.
//!
//! Dense GGUF types keep their element type. Block-quantized GGML types stay
//! opaque `Uint8` rows of whole blocks, `[leading..., row_bytes]`, so device
//! memory holds exactly the file's bytes and kernels decode them in place.

use std::collections::BTreeSet;

use mirtal::{DType, TensorRange};
use models::weights::{DenseDType, TensorDType, TensorInfo};

use crate::engine::{Error, Result};

/// The ranges of one GGUF file's `tensors`, its share of its model's
/// catalog.
pub(super) fn ranges(tensors: &[TensorInfo]) -> Result<Vec<TensorRange>> {
    tensors.iter().map(range).collect()
}

fn range(tensor: &TensorInfo) -> Result<TensorRange> {
    let mut shape = tensor
        .shape
        .iter()
        .map(|&dimension| u64::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let dtype = match &tensor.dtype {
        TensorDType::Dense(DenseDType::F32) => DType::Float32,
        TensorDType::Dense(DenseDType::F16) => DType::Float16,
        TensorDType::Dense(DenseDType::Bf16) => DType::Bfloat16,
        // Raw E4M3 bytes, as the FP8 projections read `SafeTensors` FP8.
        TensorDType::Dense(DenseDType::F8E4M3) => DType::Uint8,
        TensorDType::Dense(_) => return Err(invalid(tensor, "a dense type Metal does not load")),
        TensorDType::Ggml(ggml) => {
            let input = shape.pop().ok_or_else(|| invalid(tensor, "no dimensions"))?;
            shape.push(ggml.row_bytes(input)?);
            DType::Uint8
        },
    };
    Ok(TensorRange::new(tensor.name.clone(), tensor.payload_start()?, dtype, shape)?)
}

fn invalid(tensor: &TensorInfo, reason: &str) -> Error {
    Error::InvalidModel(format!("GGUF tensor `{}` has {reason}", tensor.name))
}

/// GGUF stores norms in F32 and some weights in F16; the Metal runtime
/// computes in BF16, as the CUDA loader does for the same files, except for
/// the binding plan's `full_precision` sources (quantized weights' scales,
/// a GGML output scale, a sigmoid router's correction bias).
pub(super) fn casts_to_bf16(range: &TensorRange, full_precision: &BTreeSet<String>) -> bool {
    matches!(range.dtype(), DType::Float32 | DType::Float16)
        && !full_precision.contains(range.name())
}
