mod dense;
mod gated;
mod pair;
mod reduce;
#[cfg(all(test, target_os = "linux"))]
mod tests;

pub use gated::{GatedActivation, SelectedAffineGatedBf16Linear};
use mircuda::DeviceBuffer;
pub use pair::{AffineQuantizedPairTensors, SelectedAffinePairBf16Linear};
pub use reduce::SelectedAffineReduceBf16Linear;

use super::{AffineQuantizedTensors, quantized::validate_shape};
use crate::{CudaTensor, Error, Result};

fn validate_bank(
    tensors: AffineQuantizedTensors<'_>,
    weight_shape: &[usize],
    group_shape: &[usize],
) -> Result<()> {
    validate_shape(tensors.weight, weight_shape.to_vec())?;
    validate_shape(tensors.scales, group_shape.to_vec())?;
    validate_shape(tensors.biases, group_shape.to_vec())
}

fn u32_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<u32>> {
    tensor.as_u32().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "U32",
    })
}
pub use dense::DenseExpertWeights;
pub(in crate::backend) use dense::SelectedDenseMoeBf16;
