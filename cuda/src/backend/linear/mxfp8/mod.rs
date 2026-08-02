mod candidate;
mod execution;
mod moe;
#[cfg(all(test, target_os = "linux"))]
mod tests;
mod tuning;
mod weight;
mod weights;

use std::sync::{Arc, OnceLock};

use mircuda::{DeviceBuffer, MxFp8Embedding, MxFp8Gathered, MxFp8Spec};
use models::weights::{BlockProjectionLayout, BlockQuantization, TensorBinding};
pub use moe::MxFp8GatheredMoeBf16;
pub use weights::MxFp8ExpertWeights;

use crate::{CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result};

#[derive(Clone, Debug)]
/// Validated model-owned OCP MXFP8 checkpoint tensors.
pub struct MxFp8CheckpointWeight {
    weight: CudaTensor,
    scales: CudaTensor,
    bias: Option<CudaTensor>,
    input_features: usize,
    output_features: usize,
    layout: BlockProjectionLayout,
    swizzled_scales: Arc<OnceLock<DeviceBuffer<u8>>>,
}

#[derive(Debug)]
/// Prepared BF16-input projection retaining packed MXFP8 checkpoint tensors.
pub struct MxFp8Bf16Linear {
    operation: candidate::Candidate,
    stream: mircuda::Stream,
    pool: mircuda::MemoryPool,
    spec: MxFp8Spec,
}

#[derive(Debug)]
/// Prepared selected-row MXFP8 embedding lookup.
pub struct MxFp8EmbeddingLookup {
    operation: MxFp8Embedding,
    stream: mircuda::Stream,
    weight: MxFp8CheckpointWeight,
}

#[derive(Debug)]
/// Prepared gathered projection retaining an MXFP8 matrix bank.
pub struct MxFp8GatheredBf16Linear {
    operation: MxFp8Gathered,
    stream: mircuda::Stream,
    matrices: usize,
    input_features: usize,
    output_features: usize,
    has_bias: bool,
}

fn projection_shape(
    binding: &TensorBinding,
) -> Result<(BlockProjectionLayout, Vec<usize>, usize, usize)> {
    let (layout, prefix, output, input) =
        match (binding.block_projection_layout(), binding.logical_shape.as_deref()) {
            (Some(BlockProjectionLayout::Matrix), Some([output, input])) => {
                (BlockProjectionLayout::Matrix, Vec::new(), *output, *input)
            },
            (
                Some(layout @ BlockProjectionLayout::MatrixBank { matrices }),
                Some([actual, output, input]),
            ) if matrices == *actual => (layout, vec![matrices], *output, *input),
            (
                Some(
                    layout @ BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true },
                ),
                Some([actual, output, input]),
            ) if experts == *actual => (layout, vec![experts], *output, *input),
            _ => {
                return Err(unsupported(binding, "requires an ordinary or gathered matrix layout"));
            },
        };
    if !input.is_multiple_of(BlockQuantization::MXFP8.block_size) {
        return Err(unsupported(binding, "input width is not a complete MXFP8 block"));
    }
    Ok((layout, prefix, output, input))
}

fn tensor(
    tensors: &CudaTensorSet,
    name: &str,
    expected_dtype: CudaTensorDType,
    expected: &'static str,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
    if tensor.dtype() != expected_dtype {
        return Err(dtype(tensor, expected));
    }
    Ok(tensor.clone())
}

fn require_shape(tensor: &CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.into(),
            actual: tensor.shape().into(),
        })
    }
}

fn dtype(tensor: &CudaTensor, expected: &'static str) -> Error {
    Error::DTypeMismatch { name: tensor.name().into(), expected }
}

fn unsupported(binding: &TensorBinding, reason: &str) -> Error {
    Error::UnsupportedDecoderLayer(format!("{}: {reason}", binding.source))
}
