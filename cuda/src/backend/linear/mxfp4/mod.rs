use mircuda::DeviceBuffer;
use models::weights::{BlockProjectionLayout, BlockQuantization, TensorBinding};

use crate::{
    CudaTensor, CudaTensorSet, Error, Result,
    kernels::{MxFp4GatheredSpec, MxFp4Spec},
};

mod execution;
mod load;
mod moe;
mod weights;

pub use moe::MxFp4GatheredMoeBf16;
pub use weights::MxFp4ExpertWeights;

#[derive(Clone, Debug)]
/// Validated model-owned OCP MXFP4 checkpoint matrix.
pub struct MxFp4CheckpointWeight {
    packed: DeviceBuffer<u8>,
    scales: CudaTensor,
    bias: Option<CudaTensor>,
    input_features: usize,
    output_features: usize,
    layout: BlockProjectionLayout,
}

#[derive(Debug)]
/// Prepared BF16-input projection retaining packed MXFP4 checkpoint tensors.
pub struct MxFp4Bf16Linear {
    operation: crate::kernels::MxFp4Linear,
    stream: mircuda::Stream,
    spec: MxFp4Spec,
    has_bias: bool,
}

#[derive(Debug)]
/// Prepared gathered projection retaining a packed MXFP4 matrix bank.
pub struct MxFp4GatheredBf16Linear {
    operation: crate::kernels::MxFp4GatheredLinear,
    stream: mircuda::Stream,
    spec: MxFp4GatheredSpec,
    has_bias: bool,
}

#[derive(Debug)]
/// Prepared selected-row OCP MXFP4 embedding lookup.
pub struct MxFp4EmbeddingLookup {
    operation: crate::kernels::MxFp4Embedding,
    stream: mircuda::Stream,
    weight: MxFp4CheckpointWeight,
}

impl MxFp4CheckpointWeight {
    pub fn prepare(&self, backend: &super::CudaBackend, tokens: usize) -> Result<MxFp4Bf16Linear> {
        if self.layout != BlockProjectionLayout::Matrix {
            return Err(Error::InvalidExecutionPlan(
                "MXFP4 matrix bank cannot use ordinary projection",
            ));
        }
        let spec = MxFp4Spec::new(tokens, self.input_features, self.output_features)?;
        Ok(MxFp4Bf16Linear {
            operation: crate::kernels::MxFp4Linear::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
            spec,
            has_bias: self.bias.is_some(),
        })
    }

    pub(crate) fn validate(&self, input: usize, output: usize) -> Result<()> {
        if self.layout == BlockProjectionLayout::Matrix
            && self.input_features == input
            && self.output_features == output
        {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("MXFP4 checkpoint geometry differs"))
        }
    }

    fn validate_bank(&self, matrices: usize, input: usize, output: usize) -> Result<()> {
        if self.layout == (BlockProjectionLayout::MatrixBank { matrices })
            && self.input_features == input
            && self.output_features == output
        {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("MXFP4 matrix-bank geometry differs"))
        }
    }

    fn validate_interleaved_bank(
        &self,
        experts: usize,
        input: usize,
        intermediate: usize,
    ) -> Result<()> {
        if self.layout == (BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true })
            && self.input_features == input
            && self.output_features
                == intermediate
                    .checked_mul(2)
                    .ok_or(Error::InvalidDecoderKernel("MXFP4 gate/up size overflow"))?
        {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("MXFP4 gate/up-bank geometry differs"))
        }
    }

    pub fn prepare_gathered(
        &self,
        backend: &super::CudaBackend,
        assignments: usize,
    ) -> Result<MxFp4GatheredBf16Linear> {
        self.prepare_gathered_routed(backend, assignments, 1)
    }

    fn prepare_gathered_warps(
        &self,
        backend: &super::CudaBackend,
        assignments: usize,
        warps_per_block: usize,
    ) -> Result<MxFp4GatheredBf16Linear> {
        self.prepare_gathered_routed_warps(backend, assignments, 1, warps_per_block)
    }

    pub(in crate::backend) fn prepare_gathered_routed(
        &self,
        backend: &super::CudaBackend,
        input_rows: usize,
        selections_per_input: usize,
    ) -> Result<MxFp4GatheredBf16Linear> {
        self.prepare_gathered_routed_warps(backend, input_rows, selections_per_input, 8)
    }

    pub(in crate::backend) fn prepare_gathered_routed_warps(
        &self,
        backend: &super::CudaBackend,
        input_rows: usize,
        selections_per_input: usize,
        warps_per_block: usize,
    ) -> Result<MxFp4GatheredBf16Linear> {
        let matrices = match self.layout {
            BlockProjectionLayout::MatrixBank { matrices }
            | BlockProjectionLayout::FusedGateUpBank { experts: matrices, .. } => matrices,
            BlockProjectionLayout::Matrix => {
                return Err(Error::InvalidExecutionPlan(
                    "ordinary MXFP4 matrix cannot use gathered projection",
                ));
            },
        };
        let spec = MxFp4GatheredSpec::new_routed(
            input_rows,
            selections_per_input,
            matrices,
            self.input_features,
            self.output_features,
        )?;
        Ok(MxFp4GatheredBf16Linear {
            operation: crate::kernels::MxFp4GatheredLinear::compile_warps(
                &backend.inner.compiler,
                spec,
                warps_per_block,
            )?,
            stream: backend.inner.stream.clone(),
            spec,
            has_bias: self.bias.is_some(),
        })
    }

    pub fn prepare_embedding(
        &self,
        backend: &super::CudaBackend,
        output_scale: f32,
    ) -> Result<MxFp4EmbeddingLookup> {
        if self.layout != BlockProjectionLayout::Matrix || self.bias.is_some() {
            return Err(Error::InvalidExecutionPlan("MXFP4 embedding cannot have output bias"));
        }
        let spec = crate::kernels::MxFp4EmbeddingSpec::new(
            self.output_features,
            self.input_features,
            output_scale,
        )?;
        Ok(MxFp4EmbeddingLookup {
            operation: crate::kernels::MxFp4Embedding::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
            weight: self.clone(),
        })
    }
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
    if !input.is_multiple_of(BlockQuantization::MXFP4.block_size) {
        return Err(unsupported(binding, "input width is not a complete MXFP4 block"));
    }
    Ok((layout, prefix, output, input))
}

fn unsupported(binding: &TensorBinding, reason: &str) -> Error {
    Error::UnsupportedDecoderLayer(format!("{}: {reason}", binding.source))
}

fn buffer<'a>(
    value: Option<&'a DeviceBuffer<u8>>,
    tensor: &CudaTensor,
    expected: &'static str,
) -> Result<&'a DeviceBuffer<u8>> {
    value.ok_or_else(|| Error::DTypeMismatch { name: tensor.name().into(), expected })
}
