use std::sync::{Arc, OnceLock};

use mircuda::{MxFp8Embedding, MxFp8EmbeddingSpec, MxFp8Gathered, MxFp8GatheredSpec, MxFp8Spec};
use models::weights::{BlockProjectionLayout, BlockQuantization, TensorBinding, TensorStorage};

use super::{
    MxFp8Bf16Linear, MxFp8CheckpointWeight, MxFp8EmbeddingLookup, MxFp8GatheredBf16Linear,
    projection_shape, require_shape, tensor, tuning, unsupported,
};
use crate::{CudaBackend, CudaTensorDType, CudaTensorSet, Error, Result};

impl MxFp8CheckpointWeight {
    /// Loads separate U32 E4M3 words and U8 E8M0 scales without conversion.
    pub fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        let TensorStorage::BlockQuantized {
            format,
            scales,
            global_scale: None,
            input_scale: None,
            bias,
            packing: _,
        } = &binding.storage
        else {
            return Err(unsupported(binding, "requires self-contained MXFP8 storage"));
        };
        if *format != BlockQuantization::MXFP8 {
            return Err(unsupported(binding, "does not match the native CUDA MXFP8 contract"));
        }
        let (layout, prefix, output_features, input_features) = projection_shape(binding)?;
        let mut weight_shape = prefix.clone();
        weight_shape.extend([output_features, input_features / 4]);
        let weight = tensor(tensors, &binding.source, CudaTensorDType::U32, "U32")?;
        require_shape(&weight, &weight_shape)?;
        let mut scale_shape = prefix.clone();
        scale_shape.extend([output_features, input_features / 32]);
        let scales = tensor(tensors, scales, CudaTensorDType::U8, "U8")?;
        require_shape(&scales, &scale_shape)?;
        let mut bias_shape = prefix;
        bias_shape.push(output_features);
        let bias = bias
            .as_deref()
            .map(|name| tensor(tensors, name, CudaTensorDType::Bf16, "BF16"))
            .transpose()?;
        if let Some(bias) = &bias {
            require_shape(bias, &bias_shape)?;
        }
        Ok(Self {
            weight,
            scales,
            bias,
            input_features,
            output_features,
            layout,
            swizzled_scales: Arc::new(OnceLock::new()),
        })
    }

    /// Compiles a fixed-token operation for this checkpoint matrix.
    pub fn prepare(&self, backend: &CudaBackend, tokens: usize) -> Result<MxFp8Bf16Linear> {
        if self.layout != BlockProjectionLayout::Matrix || self.bias.is_some() {
            return Err(Error::InvalidExecutionPlan(
                "MXFP8 ordinary projection requires a matrix without bias",
            ));
        }
        let spec = MxFp8Spec::new(tokens, self.input_features, self.output_features)?;
        let operation = tuning::prepare(backend, spec, self)?;
        Ok(MxFp8Bf16Linear {
            operation,
            stream: backend.inner.stream.clone(),
            pool: backend.inner.pool.clone(),
            spec,
        })
    }

    pub fn prepare_gathered_routed(
        &self,
        backend: &CudaBackend,
        input_rows: usize,
        selections_per_input: usize,
    ) -> Result<MxFp8GatheredBf16Linear> {
        self.prepare_gathered_routed_warps(backend, input_rows, selections_per_input, 8)
    }

    pub(super) fn prepare_gathered_routed_warps(
        &self,
        backend: &CudaBackend,
        input_rows: usize,
        selections_per_input: usize,
        warps_per_block: usize,
    ) -> Result<MxFp8GatheredBf16Linear> {
        let matrices = match self.layout {
            BlockProjectionLayout::MatrixBank { matrices }
            | BlockProjectionLayout::FusedGateUpBank { experts: matrices, .. } => matrices,
            BlockProjectionLayout::Matrix => {
                return Err(Error::InvalidExecutionPlan(
                    "ordinary MXFP8 matrix cannot use gathered projection",
                ));
            },
        };
        let spec = MxFp8GatheredSpec::new_routed(
            input_rows,
            selections_per_input,
            matrices,
            self.input_features,
            self.output_features,
        )?;
        Ok(MxFp8GatheredBf16Linear {
            operation: MxFp8Gathered::compile_warps(
                &backend.inner.compiler,
                spec,
                u32::try_from(warps_per_block)?,
            )?,
            stream: backend.inner.stream.clone(),
            matrices,
            input_features: self.input_features,
            output_features: self.output_features,
            has_bias: self.bias.is_some(),
        })
    }

    pub(super) fn prepare_gathered_warps(
        &self,
        backend: &CudaBackend,
        assignments: usize,
        warps_per_block: usize,
    ) -> Result<MxFp8GatheredBf16Linear> {
        self.prepare_gathered_routed_warps(backend, assignments, 1, warps_per_block)
    }

    pub(super) fn validate_bank(
        &self,
        matrices: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<()> {
        if self.layout == (BlockProjectionLayout::MatrixBank { matrices })
            && self.input_features == input_features
            && self.output_features == output_features
        {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("MXFP8 matrix-bank geometry differs"))
        }
    }

    pub(super) fn validate_interleaved_bank(
        &self,
        experts: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<()> {
        if self.layout == (BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true })
            && self.input_features == input_features
            && self.output_features == output_features * 2
        {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("interleaved MXFP8 gate/up geometry differs"))
        }
    }

    #[must_use]
    /// Returns the logical input width.
    pub const fn input_features(&self) -> usize {
        self.input_features
    }

    #[must_use]
    /// Returns the logical output width.
    pub const fn output_features(&self) -> usize {
        self.output_features
    }

    /// Validates this matrix against one logical projection geometry.
    pub fn validate(&self, input_features: usize, output_features: usize) -> Result<()> {
        if self.input_features == input_features && self.output_features == output_features {
            Ok(())
        } else {
            Err(Error::InvalidExecutionPlan("MXFP8 checkpoint geometry differs"))
        }
    }

    /// Prepares selected-row dequantization for an embedding matrix.
    pub fn prepare_embedding(
        &self,
        backend: &CudaBackend,
        output_scale: f32,
    ) -> Result<MxFp8EmbeddingLookup> {
        if self.layout != BlockProjectionLayout::Matrix || self.bias.is_some() {
            return Err(Error::InvalidExecutionPlan(
                "MXFP8 embedding requires an ordinary matrix without bias",
            ));
        }
        let spec =
            MxFp8EmbeddingSpec::new(self.output_features, self.input_features, output_scale)?;
        Ok(MxFp8EmbeddingLookup {
            operation: MxFp8Embedding::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
            weight: self.clone(),
        })
    }
}
