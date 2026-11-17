use mircuda::{DeviceBuffer, bf16};

use super::{CudaTensor, CudaTensorSet, TensorStorage};
use crate::{Error, Result};

impl CudaTensor {
    pub(crate) fn from_f8_e4m3(name: String, shape: Vec<usize>, buffer: DeviceBuffer<u8>) -> Self {
        Self {
            name,
            shape,
            storage: TensorStorage::F8E4M3(buffer),
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
    /// Adds `offset` to every value of a BF16 tensor in place, rounded to
    /// BF16: norms that scale by `1 + weight` (Gemma) then run as plain RMS
    /// norms.
    pub(crate) fn add_bf16(&self, name: &str, offset: f64) -> Result<()> {
        #[expect(clippy::cast_possible_truncation, reason = "a norm weight offset such as 1")]
        let offset = offset as f32;
        let tensor = self.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
        let mut device = tensor
            .as_bf16()
            .cloned()
            .ok_or_else(|| Error::DTypeMismatch { name: name.into(), expected: "BF16" })?;
        let shifted = self
            .read_bf16(name)?
            .iter()
            .map(|value| bf16::from_f32(value.to_f32() + offset))
            .collect::<Vec<_>>();
        let mut host = self.context.allocate_pinned::<bf16>(shifted.len())?;
        host.copy_from_slice(&shifted)?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        self.stream.synchronize()?;
        Ok(())
    }
}
