mod bank;
mod linear;
mod native;
#[cfg(all(test, target_os = "linux"))]
mod reference;
mod scale;
mod selected;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod weight_only;

pub(in crate::backend) use bank::MarlinNvFp4Bank;
pub use bank::{NvFp4ExpertBank, NvFp4ExpertBankConfig, NvFp4ExpertSource};
use mircuda::DeviceBuffer;
pub use scale::NvFp4ScaleMode;
pub use selected::{
    BucketedNvFp4MoeBf16, DirectNvFp4MoeBf16, GroupedNvFp4MoeBf16, HybridNvFp4MoeBf16,
    SelectedNvFp4LinearBf16, SelectedNvFp4MoeBf16, SelectedNvFp4TensorCoreMoeBf16,
};
pub(in crate::backend) use selected::{
    BucketedNvFp4Scratch, BucketedNvFp4ScratchConfig, MarlinNvFp4MoeBf16, MarlinNvFp4Scratch,
    MarlinNvFp4ScratchConfig, MarlinNvFp4UngatedMoeBf16, MarlinRouteBlock,
    SelectedNvFp4WeightOnlyTensorCoreMoeBf16, TiledSelectedNvFp4MoeBf16,
};
pub(in crate::backend) use weight_only::MarlinNvFp4Bf16Linear;
pub use weight_only::{NvFp4WeightOnlyBf16Linear, NvFp4WeightOnlyWeight};

use super::CudaBackend;
use crate::{CudaTensor, Error, Result, kernels::NvFp4Spec};

/// Model-declared geometry of one NVIDIA FP4 checkpoint projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4Config {
    pub input_features: usize,
    pub output_features: usize,
}

impl NvFp4Config {
    #[must_use]
    pub const fn new(input_features: usize, output_features: usize) -> Self {
        Self { input_features, output_features }
    }
}

/// Checkpoint tensors forming one ModelOpt-compatible NVFP4 weight.
#[derive(Clone, Copy)]
pub struct NvFp4Tensors<'a> {
    pub weight: &'a CudaTensor,
    pub weight_scale: &'a CudaTensor,
    pub weight_scale_2: &'a CudaTensor,
    pub input_scale: &'a CudaTensor,
    pub scale_mode: NvFp4ScaleMode,
}

/// Native W4A4 Tensor Core projection with persistent accelerator scratch.
#[derive(Debug)]
pub struct NvFp4Bf16Linear {
    execution: native::NativeNvFp4,
}

/// Native projections sharing one BF16-to-FP4 activation quantization.
#[derive(Debug)]
pub struct NvFp4Bf16Pack<const N: usize> {
    execution: native::NativeNvFp4Pack<N>,
}

/// Device-ready NVFP4 weight shared by decode and prefill plans.
#[derive(Clone, Debug)]
pub struct NvFp4LinearWeight {
    prepared: native::NativeNvFp4Weight,
}

impl NvFp4LinearWeight {
    #[must_use]
    pub const fn config(&self) -> NvFp4Config {
        self.prepared.config
    }
}

impl CudaBackend {
    pub fn prepare_nvfp4_linear_weight(
        &self,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<NvFp4LinearWeight> {
        Ok(NvFp4LinearWeight {
            prepared: native::NativeNvFp4Weight::prepare(self, config, tensors)?,
        })
    }

    /// Concatenates output rows of compatible NVFP4 projections once during
    /// model preparation.
    pub fn pack_nvfp4_linear_weights<const N: usize>(
        &self,
        weights: [&NvFp4LinearWeight; N],
    ) -> Result<NvFp4LinearWeight> {
        Ok(NvFp4LinearWeight {
            prepared: native::NativeNvFp4Weight::pack(
                self,
                weights.map(|weight| &weight.prepared),
            )?,
        })
    }
}

fn validate_tensors(spec: NvFp4Spec, tensors: NvFp4Tensors<'_>) -> Result<()> {
    validate_shape(tensors.weight, &[spec.output_features, spec.input_features / 2])?;
    validate_shape(tensors.weight_scale, &[spec.output_features, spec.input_features / 16])?;
    validate_scalar(tensors.weight_scale_2)?;
    validate_scalar(tensors.input_scale)
}

fn validate_scalar(tensor: &CudaTensor) -> Result<()> {
    if tensor.shape().is_empty() || tensor.shape() == [1] {
        Ok(())
    } else {
        Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: vec![1],
            actual: tensor.shape().to_vec(),
        })
    }
}

fn validate_shape(tensor: &CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        })
    }
}

fn u8_tensor<'a>(tensor: &'a CudaTensor, expected: &'static str) -> Result<&'a DeviceBuffer<u8>> {
    let value = if expected == "F8_E4M3" {
        tensor.as_f8_e4m3()
    } else {
        tensor.as_u8()
    };
    value.ok_or_else(|| Error::DTypeMismatch { name: tensor.name().into(), expected })
}

fn f32_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<f32>> {
    tensor.as_f32().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "F32",
    })
}

fn output_elements(tokens: usize, output_features: usize) -> Result<usize> {
    tokens.checked_mul(output_features).ok_or_else(|| Error::InvalidTensorSize {
        name: "NVFP4 linear output".into(),
        expected: usize::MAX,
        actual: 0,
    })
}
