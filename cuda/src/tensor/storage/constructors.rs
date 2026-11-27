use mircuda::{DeviceBuffer, bf16};

use super::{CudaTensor, CudaTensorSet, TensorStorage};
use crate::{Error, Result, kernels::GgmlBlocks};

impl CudaTensor {
    pub(crate) fn from_f8_e4m3(name: String, shape: Vec<usize>, buffer: DeviceBuffer<u8>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::F8E4M3(buffer),
        }
    }

    pub(crate) fn from_ggml_blocks(name: String, shape: Vec<usize>, blocks: GgmlBlocks) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::GgmlBlocks(blocks),
        }
    }

    /// GGML blocks in their padded allocation; other U8 storage is not.
    #[must_use]
    pub fn as_ggml_blocks(&self) -> Option<&GgmlBlocks> {
        match &self.storage {
            TensorStorage::GgmlBlocks(blocks) => Some(blocks),
            _ => None,
        }
    }

    pub(crate) fn from_f32(name: String, shape: Vec<usize>, buffer: DeviceBuffer<f32>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::F32(buffer),
        }
    }
}

impl CudaTensorSet {
    /// A set of `tensors` on `stream`, for tests that build their own.
    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn from_tensors(
        context: &mircuda::Context,
        stream: &mircuda::Stream,
        tensors: Vec<CudaTensor>,
    ) -> Self {
        Self {
            tensors: tensors.into_iter().map(|tensor| (tensor.name.clone(), tensor)).collect(),
            context: context.clone(),
            stream: stream.clone(),
        }
    }

    /// Replaces the tensor of the same name, such as a reordered copy.
    pub(crate) fn replace(&mut self, tensor: CudaTensor) -> Result<()> {
        let slot = self
            .tensors
            .get_mut(&tensor.name)
            .ok_or_else(|| Error::MissingTensor(tensor.name.clone()))?;
        *slot = tensor;
        Ok(())
    }
}

impl CudaTensorSet {
    /// Multiplies a BF16 tensor by `factor` in place, at load.
    pub(crate) fn scale_bf16(&self, name: &str, factor: f32) -> Result<()> {
        self.map_bf16(name, |value| value * factor)
    }

    /// Adds `offset` to every value of a BF16 tensor in place, rounded to
    /// BF16: norms that scale by `1 + weight` (Gemma) then run as plain RMS
    /// norms.
    pub(crate) fn add_bf16(&self, name: &str, offset: f64) -> Result<()> {
        #[expect(clippy::cast_possible_truncation, reason = "a norm weight offset such as 1")]
        let offset = offset as f32;
        self.map_bf16(name, |value| value + offset)
    }

    fn map_bf16(&self, name: &str, map: impl Fn(f32) -> f32) -> Result<()> {
        let tensor = self.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
        let mut device = tensor
            .as_bf16()
            .cloned()
            .ok_or_else(|| Error::DTypeMismatch { name: name.into(), expected: "BF16" })?;
        let mapped = self
            .read_bf16(name)?
            .iter()
            .map(|value| bf16::from_f32(map(value.to_f32())))
            .collect::<Vec<_>>();
        let mut host = self.context.allocate_pinned::<bf16>(mapped.len())?;
        host.copy_from_slice(&mapped)?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        self.stream.synchronize()?;
        Ok(())
    }
}
