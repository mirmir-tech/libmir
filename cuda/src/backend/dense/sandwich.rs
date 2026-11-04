use mircuda::{DeviceBuffer, bf16};

use super::{CudaBackend, DenseSandwichWeights};
use crate::{CudaTensor, Error, Result, RmsNormBf16};

/// Device-resident [`DenseSandwichWeights`] retained by a layer.
#[derive(Clone)]
pub(in crate::backend::dense) struct DenseSandwichOwned {
    attention_output_norm: CudaTensor,
    feed_forward_output_norm: CudaTensor,
    layer_scalar: CudaTensor,
}

impl DenseSandwichOwned {
    pub(in crate::backend::dense) fn new(weights: DenseSandwichWeights<'_>) -> Result<Self> {
        scalar(weights.layer_scalar)?;
        Ok(Self {
            attention_output_norm: weights.attention_output_norm.clone(),
            feed_forward_output_norm: weights.feed_forward_output_norm.clone(),
            layer_scalar: weights.layer_scalar.clone(),
        })
    }

    pub(in crate::backend::dense) const fn borrow(&self) -> DenseSandwichWeights<'_> {
        DenseSandwichWeights {
            attention_output_norm: &self.attention_output_norm,
            feed_forward_output_norm: &self.feed_forward_output_norm,
            layer_scalar: &self.layer_scalar,
        }
    }
}

/// Output norms of both sub-blocks for a fixed number of rows.
pub(in crate::backend::dense) struct SandwichNorms {
    pub(in crate::backend::dense) attention_output: RmsNormBf16,
    pub(in crate::backend::dense) feed_forward_output: RmsNormBf16,
}

impl SandwichNorms {
    pub(in crate::backend::dense) fn new(
        backend: &CudaBackend,
        rows: usize,
        hidden: usize,
        epsilon: f32,
    ) -> Result<Self> {
        Ok(Self {
            attention_output: RmsNormBf16::new(backend, rows, hidden, epsilon)?,
            feed_forward_output: RmsNormBf16::new(backend, rows, hidden, epsilon)?,
        })
    }
}

pub(in crate::backend::dense) fn scalar(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    if tensor.shape() != [1] {
        return Err(Error::InvalidDecoderKernel("layer scalar must contain one BF16 value"));
    }
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
