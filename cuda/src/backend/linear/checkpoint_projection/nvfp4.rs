use models::weights::{BlockQuantization, TensorBinding, TensorStorage};

use crate::{
    CudaBackend, CudaTensorSet, Error, NvFp4Config, NvFp4LinearWeight, NvFp4Tensors,
    NvFp4WeightOnlyWeight, Result,
};

pub(super) fn load_native(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
) -> Result<NvFp4LinearWeight> {
    let TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        ..
    } = &binding.storage
    else {
        return Err(Error::InvalidNvFp4("projection has no complete NVFP4 binding"));
    };
    let [output, input] = binding.logical_shape.as_deref().ok_or_else(|| {
        Error::UnsupportedDecoderLayer(format!(
            "NVFP4 projection has no logical matrix shape: {}",
            binding.source
        ))
    })?
    else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "NVFP4 projection is not a matrix: {}",
            binding.source
        )));
    };
    let get = |name: &str| tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()));
    backend.prepare_nvfp4_linear_weight(
        NvFp4Config::new(*input, *output),
        NvFp4Tensors {
            weight: get(&binding.source)?,
            weight_scale: get(scales)?,
            weight_scale_2: get(global_scale)?,
            input_scale: get(input_scale)?,
        },
    )
}

pub(super) fn load_weight_only(
    tensor_set: &CudaTensorSet,
    binding: &TensorBinding,
) -> Result<NvFp4WeightOnlyWeight> {
    let (config, tensors) = tensors(binding, tensor_set)?;
    NvFp4WeightOnlyWeight::load(config, tensors)
}

fn tensors<'a>(
    binding: &TensorBinding,
    tensors: &'a CudaTensorSet,
) -> Result<(NvFp4Config, NvFp4Tensors<'a>)> {
    let TensorStorage::BlockQuantized {
        format:
            BlockQuantization {
                format: models::weights::BlockFormat::NvFp4,
                ..
            },
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        ..
    } = &binding.storage
    else {
        return Err(Error::InvalidNvFp4("projection has no complete NVFP4 binding"));
    };
    let [output, input] = binding.logical_shape.as_deref().ok_or_else(|| {
        Error::UnsupportedDecoderLayer(format!(
            "NVFP4 projection has no logical matrix shape: {}",
            binding.source
        ))
    })?
    else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "NVFP4 projection is not a matrix: {}",
            binding.source
        )));
    };
    let get = |name: &str| tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()));
    Ok((
        NvFp4Config::new(*input, *output),
        NvFp4Tensors {
            weight: get(&binding.source)?,
            weight_scale: get(scales)?,
            weight_scale_2: get(global_scale)?,
            input_scale: get(input_scale)?,
        },
    ))
}
