use models::weights::{
    BindingTransform, BlockActivationMode, BlockFormat, TensorBinding, TensorStorage,
};

use super::{CheckpointProjectionWeight, nvfp4};
use crate::{
    AffineQuantizedWeight, CudaBackend, CudaTensorSet, DirectFp8CheckpointWeight, Error,
    MxFp4CheckpointWeight, MxFp8CheckpointWeight, PackedIntegerWeight, Result,
};

impl CheckpointProjectionWeight {
    pub(crate) fn load_binding_prepared(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
    ) -> Result<Self> {
        if let TensorStorage::BlockQuantized { format, .. } = binding.storage
            && format.format == BlockFormat::NvFp4
        {
            return match format.activation_mode {
                BlockActivationMode::WeightOnly => {
                    nvfp4::load_weight_only(tensors, binding).map(Self::NvFp4WeightOnly)
                },
                BlockActivationMode::WeightAndActivation => {
                    nvfp4::load_native(backend, tensors, binding).map(Self::NvFp4)
                },
            };
        }
        Self::load_binding(tensors, binding)
    }

    pub(crate) fn load_binding(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<Self> {
        match &binding.storage {
            TensorStorage::AffineQuantized { .. } => {
                AffineQuantizedWeight::load_binding(tensors, binding).map(Self::Affine)
            },
            TensorStorage::Dense { bias: None, .. }
                if !binding.transforms.contains(&BindingTransform::Transpose) =>
            {
                tensors
                    .get(&binding.source)
                    .cloned()
                    .ok_or_else(|| Error::MissingTensor(binding.source.clone()))
                    .map(Self::Dense)
            },
            TensorStorage::Dense { bias: Some(_), .. } => Err(Error::UnsupportedDecoderLayer(
                format!("CUDA dense checkpoint projection has a bias: {}", binding.source),
            )),
            TensorStorage::Dense { .. } => Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA dense checkpoint projection is transposed: {}",
                binding.source
            ))),
            TensorStorage::Float8 { .. } => {
                DirectFp8CheckpointWeight::load_binding(tensors, binding).map(Self::DirectFp8)
            },
            TensorStorage::BlockQuantized { format, .. } if format.format == BlockFormat::MxFp4 => {
                MxFp4CheckpointWeight::load_binding(tensors, binding).map(Self::MxFp4)
            },
            TensorStorage::BlockQuantized { format, .. } if format.format == BlockFormat::MxFp8 => {
                MxFp8CheckpointWeight::load_binding(tensors, binding).map(Self::MxFp8)
            },
            TensorStorage::BlockQuantized { .. } => Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA block projection requires prepared loading: {}",
                binding.source
            ))),
            TensorStorage::PackedInt8 { .. }
            | TensorStorage::PackedInt4 { .. }
            | TensorStorage::Awq { .. }
            | TensorStorage::Gptq { .. }
            | TensorStorage::BitsAndBytes4Bit { .. } => packed(tensors, binding),
            TensorStorage::Auxiliary { .. } => Err(Error::UnsupportedDecoderLayer(format!(
                "unsupported CUDA checkpoint projection storage: {}",
                binding.source
            ))),
        }
    }
}

fn packed(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<CheckpointProjectionWeight> {
    let [output, input] = binding.logical_shape.as_deref().ok_or_else(|| {
        Error::UnsupportedDecoderLayer(format!(
            "CUDA packed integer projection has no logical matrix shape: {}",
            binding.source
        ))
    })?
    else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA packed integer projection is not a matrix: {}",
            binding.source
        )));
    };
    PackedIntegerWeight::load_binding(tensors, binding, *input, *output)
        .map(CheckpointProjectionWeight::PackedInteger)
}
