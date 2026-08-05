use super::geometry::product;
use crate::{Error, Result};

mod cached;
mod cublaslt;
mod embedding;
mod portable;
mod tensor_core;
mod weight_only_tensor_core;
pub use cached::DirectFp8CachedLinear;
pub use embedding::{DirectFp8Embedding, DirectFp8EmbeddingBatch, DirectFp8EmbeddingSpec};
pub use portable::{DirectFp8Linear, DirectFp8Scales};
pub use tensor_core::DirectFp8TensorCoreLinear;
pub use weight_only_tensor_core::DirectE5M2WeightOnlyTensorCoreLinear;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DirectFp8Activation {
    Bf16,
    DynamicE4M3Token,
    StaticE4M3Tensor,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Physical direct-checkpoint FP8 value encoding.
pub enum DirectFp8Format {
    E4M3,
    E5M2,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Scale geometry implemented by the direct checkpoint FP8 kernel.
pub enum DirectFp8Scale {
    /// One scalar applies to the complete matrix.
    Tensor,
    /// One scalar applies to each output row.
    OutputChannel,
    /// Two-dimensional grid with explicit output-row and input-column blocks.
    BlockGrid {
        output_groups: usize,
        input_groups: usize,
        output_block_size: usize,
        input_block_size: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Fixed direct FP8 weight and BF16 activation projection geometry.
pub struct DirectFp8Spec {
    pub format: DirectFp8Format,
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
    pub scale: DirectFp8Scale,
    pub inverse_scale: bool,
    pub activation: DirectFp8Activation,
}

impl DirectFp8Spec {
    pub fn new(
        tokens: usize,
        input_features: usize,
        output_features: usize,
        scale: DirectFp8Scale,
        inverse_scale: bool,
        activation: DirectFp8Activation,
    ) -> Result<Self> {
        Self::new_with_format(
            DirectFp8Format::E4M3,
            tokens,
            input_features,
            output_features,
            scale,
            inverse_scale,
            activation,
        )
    }

    pub fn new_with_format(
        format: DirectFp8Format,
        tokens: usize,
        input_features: usize,
        output_features: usize,
        scale: DirectFp8Scale,
        inverse_scale: bool,
        activation: DirectFp8Activation,
    ) -> Result<Self> {
        if tokens == 0
            || input_features == 0
            || output_features == 0
            || !input_features.is_multiple_of(4)
            || (format == DirectFp8Format::E5M2 && activation != DirectFp8Activation::Bf16)
        {
            return Err(Error::InvalidDecoderKernel("invalid direct FP8 linear geometry"));
        }
        let spec = Self {
            format,
            tokens,
            input_features,
            output_features,
            scale,
            inverse_scale,
            activation,
        };
        let _ = spec.input_elements()?;
        let _ = spec.weight_elements()?;
        let _ = spec.output_elements()?;
        let _ = spec.scale_geometry()?;
        Ok(spec)
    }

    pub fn input_elements(self) -> Result<usize> {
        product(self.tokens, self.input_features)
    }

    pub fn weight_elements(self) -> Result<usize> {
        product(self.output_features, self.input_features)
    }

    pub fn output_elements(self) -> Result<usize> {
        product(self.tokens, self.output_features)
    }

    pub fn scale_elements(self) -> Result<usize> {
        let (rows, columns, _row_group, _column_group) = self.scale_geometry()?;
        product(rows, columns)
    }

    fn scale_geometry(self) -> Result<(usize, usize, usize, usize)> {
        scale_geometry(self.scale, self.output_features, self.input_features)
    }
}

fn scale_geometry(
    scale: DirectFp8Scale,
    output_features: usize,
    input_features: usize,
) -> Result<(usize, usize, usize, usize)> {
    match scale {
        DirectFp8Scale::Tensor => Ok((1, 1, output_features, input_features)),
        DirectFp8Scale::OutputChannel => Ok((output_features, 1, 1, input_features)),
        DirectFp8Scale::BlockGrid {
            output_groups,
            input_groups,
            output_block_size,
            input_block_size,
        } if output_groups > 0
            && input_groups > 0
            && output_block_size > 0
            && input_block_size.is_multiple_of(4)
            && output_groups == output_features.div_ceil(output_block_size)
            && input_groups == input_features.div_ceil(input_block_size) =>
        {
            Ok((output_groups, input_groups, output_block_size, input_block_size))
        },
        DirectFp8Scale::BlockGrid { .. } => {
            Err(Error::InvalidDecoderKernel("invalid direct FP8 scale geometry"))
        },
    }
}

#[cfg(test)]
mod tests;
pub use cublaslt::DirectFp8CublasLtLinear;
