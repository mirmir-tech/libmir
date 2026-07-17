mod block_fp8;
mod mixed;
mod nvfp4;
mod pack;
mod pair;
mod projection;
mod qmm;
mod quantized;
mod selected;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod vector;

pub use block_fp8::{BlockFp8LinearWeight, Fp8ResidualLinearWeight};
use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DeviceBuffer, Stream, bf16};
pub use mixed::Bf16Fp32Linear;
pub use nvfp4::{
    BucketedNvFp4MoeBf16, DirectNvFp4MoeBf16, GroupedNvFp4MoeBf16, HybridNvFp4MoeBf16,
    NvFp4Bf16Linear, NvFp4Bf16Pack, NvFp4Config, NvFp4ExpertBank, NvFp4ExpertBankConfig,
    NvFp4ExpertSource, NvFp4LinearWeight, NvFp4Tensors, SelectedNvFp4LinearBf16,
    SelectedNvFp4MoeBf16, SelectedNvFp4TensorCoreMoeBf16,
};
pub use pack::{Bf16LinearPack, Bf16LinearPackWeights};
pub use pair::{Bf16LinearPair, Bf16LinearPairWeights};
pub use projection::Bf16Projection;
pub use qmm::AffineQuantizedBf16Qmm;
pub use quantized::{AffineQuantizedBf16Linear, AffineQuantizedConfig, AffineQuantizedTensors};
pub use selected::{
    AffineQuantizedPairTensors, GatedActivation, SelectedAffineGatedBf16Linear,
    SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear,
};
pub use vector::Bf16VectorLinear;

use super::CudaBackend;
use crate::{CudaTensor, Error, Result};

/// Checkpoint format selected for decoder projection weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionFormat {
    Bf16,
    NvFp4,
}

/// Fixed-shape BF16 `input × weightᵀ` plan for dense checkpoint projections.
#[derive(Debug)]
pub struct Bf16Linear {
    plan: DenseMatmulPlan<bf16>,
    stream: Stream,
    tokens: usize,
    input_features: usize,
    output_features: usize,
}

impl Bf16Linear {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let spec = DenseMatmulSpec::new(tokens, output_features, input_features)?;
        Ok(Self {
            plan: DenseMatmulPlan::new(&backend.inner.context, &backend.inner.stream, spec)?,
            stream: backend.inner.stream.clone(),
            tokens,
            input_features,
            output_features,
        })
    }

    /// Enqueues one dense projection without allocation or host
    /// synchronization.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let expected_shape = [self.output_features, self.input_features];
        if weight.shape() != expected_shape {
            return Err(Error::InvalidLinearWeight {
                name: weight.name().into(),
                expected: expected_shape,
                actual: weight.shape().to_vec(),
            });
        }
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        Ok(self.plan.execute(&self.stream, input, weight, output, 1.0, 0.0)?)
    }

    /// Number of output elements required by this plan.
    pub fn output_elements(&self) -> Result<usize> {
        self.tokens
            .checked_mul(self.output_features)
            .ok_or_else(|| Error::InvalidTensorSize {
                name: "linear output".into(),
                expected: usize::MAX,
                actual: 0,
            })
    }
}
