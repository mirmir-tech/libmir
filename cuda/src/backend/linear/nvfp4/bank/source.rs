use models::weights::{BlockFormat, TensorBinding, TensorCatalog, TensorStorage};

use super::NvFp4ExpertSource;
use crate::{Error, NvFp4ScaleMode, Result};

impl<'a> NvFp4ExpertSource<'a> {
    /// Resolves one individual expert projection bound as complete `ModelOpt`
    /// NVFP4 storage.
    pub(in crate::backend) fn from_binding(
        catalog: &'a TensorCatalog,
        binding: &TensorBinding,
    ) -> Result<Self> {
        let TensorStorage::BlockQuantized {
            format,
            scales,
            global_scale: Some(global_scale),
            input_scale: Some(input_scale),
            ..
        } = &binding.storage
        else {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA individual expert is not complete NVFP4: {}",
                binding.source
            )));
        };
        if format.format != BlockFormat::NvFp4 {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA individual expert is not NVFP4: {}",
                binding.source
            )));
        }
        let get = |name: &str| catalog.get(name).ok_or_else(|| Error::MissingTensor(name.into()));
        Ok(Self {
            weight: get(&binding.source)?,
            weight_scale: get(scales)?,
            weight_scale_2: get(global_scale)?,
            input_scale: get(input_scale)?,
            scale_mode: NvFp4ScaleMode::from_names(global_scale, input_scale)?,
        })
    }
}
