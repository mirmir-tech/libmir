mod affine_projection;
mod autotune;
mod awq;
mod bitsandbytes;
mod block_fp8;
mod checkpoint_projection;
mod direct_fp8;
mod gptq;
mod mixed;
mod mxfp4;
mod mxfp8;
mod nvfp4;
mod pack;
mod packed_int8;
mod packed_integer;
mod pair;
mod projection;
mod qmm;
mod quantized;
mod selected;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod vector;
mod vendor;
mod weight;

pub(in crate::backend) use affine_projection::AffineProjection;
pub(super) use autotune::AutoBf16Plan;
pub use block_fp8::{BlockFp8LinearWeight, Fp8ResidualLinearWeight};
pub(in crate::backend) use checkpoint_projection::CheckpointProjection;
pub use checkpoint_projection::CheckpointProjectionWeight;
pub use direct_fp8::{DirectFp8Bf16Linear, DirectFp8CheckpointWeight, DirectFp8EmbeddingLookup};
use mircuda::{DenseMatmulPlan, DenseMatmulSpec, DeviceBuffer, Stream, bf16};
pub use mixed::Bf16Fp32Linear;
pub use mxfp4::{
    MxFp4Bf16Linear, MxFp4CheckpointWeight, MxFp4EmbeddingLookup, MxFp4ExpertWeights,
    MxFp4GatheredBf16Linear, MxFp4GatheredMoeBf16,
};
pub use mxfp8::{
    MxFp8Bf16Linear, MxFp8CheckpointWeight, MxFp8EmbeddingLookup, MxFp8ExpertWeights,
    MxFp8GatheredBf16Linear, MxFp8GatheredMoeBf16,
};
pub use nvfp4::{
    BucketedNvFp4MoeBf16, DirectNvFp4MoeBf16, GroupedNvFp4MoeBf16, HybridNvFp4MoeBf16,
    NvFp4Bf16Linear, NvFp4Bf16Pack, NvFp4Config, NvFp4ExpertBank, NvFp4ExpertBankConfig,
    NvFp4ExpertSource, NvFp4LinearWeight, NvFp4ScaleMode, NvFp4Tensors, NvFp4WeightOnlyBf16Linear,
    NvFp4WeightOnlyWeight, SelectedNvFp4LinearBf16, SelectedNvFp4MoeBf16,
    SelectedNvFp4TensorCoreMoeBf16,
};
pub(in crate::backend) use nvfp4::{
    BucketedNvFp4Scratch, BucketedNvFp4ScratchConfig, MarlinNvFp4Bf16Linear, MarlinNvFp4MoeBf16,
    MarlinNvFp4Scratch, MarlinNvFp4ScratchConfig, MarlinRouteBlock,
    SelectedNvFp4WeightOnlyTensorCoreMoeBf16, TiledSelectedNvFp4MoeBf16,
};
pub use pack::{Bf16LinearPack, Bf16LinearPackWeights};
pub(in crate::backend) use packed_int8::CompressedInt8Embedding;
pub use packed_int8::{CompressedInt8Bf16Linear, CompressedInt8Weight};
pub(in crate::backend) use packed_integer::embedding as packed_integer_embedding;
pub use packed_integer::{PackedIntegerBf16Linear, PackedIntegerWeight};
pub use pair::{Bf16LinearPair, Bf16LinearPairWeights};
pub use projection::Bf16Projection;
pub use qmm::AffineQuantizedBf16Qmm;
pub use quantized::{AffineQuantizedBf16Linear, AffineQuantizedConfig, AffineQuantizedTensors};
pub(in crate::backend) use selected::SelectedDenseMoeBf16;
pub use selected::{
    AffineQuantizedPairTensors, DenseExpertWeights, GatedActivation, SelectedAffineGatedBf16Linear,
    SelectedAffinePairBf16Linear, SelectedAffineReduceBf16Linear,
};
pub use vector::Bf16VectorLinear;
pub use vendor::Bf16VendorLinear;
pub use weight::AffineQuantizedWeight;

use super::CudaBackend;
use crate::{CudaTensor, CudaTensorDType, Error, Result};

/// Checkpoint format selected for decoder projection weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionFormat {
    Affine,
    Bf16,
    DirectFp8,
    MxFp4,
    MxFp8,
    PackedInteger,
    NvFp4,
}

fn validate_dense(tensor: &CudaTensor, matrices: usize, input: usize, output: usize) -> Result<()> {
    let expected = if matrices == 1 {
        vec![output, input]
    } else {
        vec![matrices, output, input]
    };
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: "BF16",
        });
    }
    if tensor.shape() != expected {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected,
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
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
        self.validate_weight(weight, false)?;
        self.execute_weight(input, weight, output)
    }

    pub(crate) fn execute_flattened(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate_weight(weight, true)?;
        self.execute_weight(input, weight, output)
    }

    fn validate_weight(&self, weight: &CudaTensor, flattened: bool) -> Result<()> {
        let shape = weight.shape();
        let input = shape.get(1..).and_then(|dimensions| {
            dimensions.iter().try_fold(1_usize, |total, value| total.checked_mul(*value))
        });
        let valid = shape.first() == Some(&self.output_features)
            && if flattened {
                input == Some(self.input_features)
            } else {
                shape == [self.output_features, self.input_features]
            };
        if !valid {
            let expected_shape = [self.output_features, self.input_features];
            return Err(Error::InvalidLinearWeight {
                name: weight.name().into(),
                expected: expected_shape,
                actual: weight.shape().to_vec(),
            });
        }
        Ok(())
    }

    fn execute_weight(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
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
