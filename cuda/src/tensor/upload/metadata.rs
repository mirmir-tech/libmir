//! Device dtype and element count of a checkpoint tensor's payload.

use models::weights::{GgmlType, TensorDType, TensorInfo};

use crate::{CudaTensorDType, Error, Result};

/// The block type of a tensor stored as quantized GGML blocks.
pub(super) const fn ggml_blocks(info: &TensorInfo) -> Option<GgmlType> {
    info.dtype.as_ggml()
}

pub(super) fn metadata(info: &TensorInfo) -> Result<(CudaTensorDType, usize)> {
    match &info.dtype {
        TensorDType::Ggml(ggml_type) => ggml_metadata(info, *ggml_type),
        TensorDType::Dense(dense) => {
            let dtype = CudaTensorDType::from_dense(dense)?;
            let elements = element_count(&info.shape, &info.name)?;
            let actual = info.payload_bytes()?;
            let expected =
                elements.checked_mul(dtype.bytes()).ok_or_else(|| Error::InvalidTensorSize {
                    name: info.name.clone(),
                    expected: usize::MAX,
                    actual,
                })?;
            if expected != actual {
                return Err(Error::InvalidTensorSize {
                    name: info.name.clone(),
                    expected,
                    actual,
                });
            }
            Ok((dtype, elements))
        },
    }
}

/// GGML blocks upload as raw bytes; the binding keeps their type and the
/// tensor keeps its logical `[rows, input]` shape.
fn ggml_metadata(info: &TensorInfo, ggml_type: GgmlType) -> Result<(CudaTensorDType, usize)> {
    let actual = info.payload_bytes()?;
    let invalid = || Error::InvalidTensorSize {
        name: info.name.clone(),
        expected: usize::MAX,
        actual,
    };
    let (&input, rows) = info.shape.split_last().ok_or_else(invalid)?;
    let row_bytes = usize::try_from(ggml_type.row_bytes(u64::try_from(input)?)?)?;
    let expected = element_count(rows, &info.name)?.checked_mul(row_bytes).ok_or_else(invalid)?;
    if expected != actual {
        return Err(Error::InvalidTensorSize {
            name: info.name.clone(),
            expected,
            actual,
        });
    }
    Ok((CudaTensorDType::U8, actual))
}

fn element_count(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| Error::InvalidTensorSize {
            name: name.into(),
            expected: usize::MAX,
            actual: 0,
        })
    })
}
