use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{TensorBinding, TensorStorage};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    kernels::{PackedInt8Launch, PackedInt8Linear, PackedInt8Spec},
};

#[derive(Clone, Debug)]
pub struct CompressedInt8Weight {
    weight: CudaTensor,
    scales: CudaTensor,
}

impl CompressedInt8Weight {
    pub(crate) fn load_binding(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let TensorStorage::PackedInt8 { scales, .. } = &binding.storage else {
            return Err(Error::InvalidQuantizedGemv(
                "binding is not a compressed-tensors INT8 weight",
            ));
        };
        let value = Self {
            weight: required(tensors, &binding.source)?,
            scales: required(tensors, scales)?,
        };
        value.validate(input, output)?;
        Ok(value)
    }

    fn validate(&self, input: usize, output: usize) -> Result<()> {
        shape(&self.weight, &[output, input / 4])?;
        shape(&self.scales, &[output, 1])?;
        dtype(&self.weight, CudaTensorDType::I32, "I32")?;
        dtype(&self.scales, CudaTensorDType::Bf16, "BF16")
    }
}

#[derive(Clone, Debug)]
pub struct CompressedInt8Bf16Linear {
    operation: PackedInt8Linear,
    stream: Stream,
    weight: CompressedInt8Weight,
}

impl CompressedInt8Bf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        weight: CompressedInt8Weight,
    ) -> Result<Self> {
        weight.validate(input, output)?;
        Ok(Self {
            operation: PackedInt8Linear::compile(
                &backend.inner.compiler,
                PackedInt8Spec::new(tokens, input, output)?,
            )?,
            stream: backend.inner.stream.clone(),
            weight,
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let weight = self.weight.weight.as_i32().ok_or_else(|| Error::DTypeMismatch {
            name: self.weight.weight.name().into(),
            expected: "I32",
        })?;
        let scales = self.weight.scales.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: self.weight.scales.name().into(),
            expected: "BF16",
        })?;
        self.operation
            .execute(&self.stream, &mut PackedInt8Launch { input, weight, scales, output })
    }
}

fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}

fn shape(tensor: &CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() != expected {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

fn dtype(tensor: &CudaTensor, expected: CudaTensorDType, name: &'static str) -> Result<()> {
    if tensor.dtype() != expected {
        return Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: name,
        });
    }
    Ok(())
}
