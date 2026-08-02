use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{TensorBinding, TensorStorage};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    kernels::{AwqLaunch, AwqLinear, AwqSpec},
};

#[derive(Clone, Debug)]
pub struct AwqWeight {
    weight: CudaTensor,
    zero_points: CudaTensor,
    scales: CudaTensor,
    group_size: usize,
}

impl AwqWeight {
    pub(crate) fn load_binding(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let TensorStorage::Awq { format, scales, zero_points } = &binding.storage else {
            return Err(Error::InvalidQuantizedGemv("binding is not an AWQ weight"));
        };
        if !format.is_gemm_w4a16() {
            return Err(Error::InvalidQuantizedGemv("CUDA requires AWQ GEMM W4A16 storage"));
        }
        let value = Self {
            weight: required(tensors, &binding.source)?,
            zero_points: required(tensors, zero_points)?,
            scales: required(tensors, scales)?,
            group_size: format.group_size,
        };
        value.validate(input, output)?;
        Ok(value)
    }

    pub(in crate::backend) fn validate(&self, input: usize, output: usize) -> Result<()> {
        if self.group_size == 0 || !input.is_multiple_of(self.group_size) {
            return Err(Error::InvalidQuantizedGemv("AWQ group geometry is invalid"));
        }
        shape(&self.weight, &[input, output / 8])?;
        shape(&self.zero_points, &[input / self.group_size, output / 8])?;
        shape(&self.scales, &[input / self.group_size, output])?;
        dtype(&self.weight, CudaTensorDType::I32, "I32")?;
        dtype(&self.zero_points, CudaTensorDType::I32, "I32")?;
        dtype(&self.scales, CudaTensorDType::F16, "F16")
    }
}

#[derive(Clone, Debug)]
pub struct AwqBf16Linear {
    operation: AwqLinear,
    stream: Stream,
    input: usize,
    output: usize,
}

impl AwqBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        weight: &AwqWeight,
    ) -> Result<Self> {
        weight.validate(input, output)?;
        Ok(Self {
            operation: AwqLinear::compile(
                &backend.inner.compiler,
                AwqSpec::new(tokens, input, output, weight.group_size)?,
            )?,
            stream: backend.inner.stream.clone(),
            input,
            output,
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &AwqWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        weight.validate(self.input, self.output)?;
        self.operation.execute(
            &self.stream,
            &mut AwqLaunch {
                input,
                weight: i32_buffer(&weight.weight)?,
                zero_points: i32_buffer(&weight.zero_points)?,
                scales: weight.scales.as_f16().ok_or_else(|| Error::DTypeMismatch {
                    name: weight.scales.name().into(),
                    expected: "F16",
                })?,
                output,
            },
        )
    }
}

fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}

fn i32_buffer(tensor: &CudaTensor) -> Result<&DeviceBuffer<i32>> {
    tensor.as_i32().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "I32",
    })
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
