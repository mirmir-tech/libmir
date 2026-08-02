use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{
    GptqBits, GptqCheckpointFormat, GptqScaleDType, TensorBinding, TensorStorage,
};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    kernels::{GptqLaunch, GptqLinear, GptqSpec},
};

#[derive(Clone, Debug)]
pub struct GptqWeight {
    weight: CudaTensor,
    zero_points: CudaTensor,
    scales: CudaTensor,
    group_indices: CudaTensor,
    group_size: usize,
    legacy: bool,
}

impl GptqWeight {
    pub(crate) fn load_binding(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let TensorStorage::Gptq {
            format,
            scales,
            zero_points,
            group_indices,
        } = &binding.storage
        else {
            return Err(Error::InvalidQuantizedGemv("binding is not a GPTQ weight"));
        };
        if format.bits != GptqBits::Four
            || format.scale_dtype != GptqScaleDType::F16
            || !format.symmetric
            || !format.is_input_packed()
        {
            return Err(Error::InvalidQuantizedGemv(
                "CUDA requires symmetric GPTQ W4A16 input packing",
            ));
        }
        let value = Self {
            weight: required(tensors, &binding.source)?,
            zero_points: required(tensors, zero_points)?,
            scales: required(tensors, scales)?,
            group_indices: required(tensors, group_indices)?,
            group_size: format.group_size,
            legacy: format.checkpoint_format == GptqCheckpointFormat::Gptq,
        };
        value.validate(input, output)?;
        Ok(value)
    }

    pub(in crate::backend) fn validate(&self, input: usize, output: usize) -> Result<()> {
        if self.group_size == 0 || !input.is_multiple_of(self.group_size) {
            return Err(Error::InvalidQuantizedGemv("GPTQ group geometry is invalid"));
        }
        shape(&self.weight, &[input / 8, output])?;
        shape(&self.zero_points, &[input / self.group_size, output / 8])?;
        shape(&self.scales, &[input / self.group_size, output])?;
        shape(&self.group_indices, &[input])?;
        dtype(&self.weight, CudaTensorDType::I32, "I32")?;
        dtype(&self.zero_points, CudaTensorDType::I32, "I32")?;
        dtype(&self.scales, CudaTensorDType::F16, "F16")?;
        dtype(&self.group_indices, CudaTensorDType::I32, "I32")
    }
}

#[derive(Clone, Debug)]
pub struct GptqBf16Linear {
    operation: GptqLinear,
    stream: Stream,
    input: usize,
    output: usize,
}

impl GptqBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        weight: &GptqWeight,
    ) -> Result<Self> {
        weight.validate(input, output)?;
        Ok(Self {
            operation: GptqLinear::compile(
                &backend.inner.compiler,
                GptqSpec::new(tokens, input, output, weight.group_size, weight.legacy)?,
            )?,
            stream: backend.inner.stream.clone(),
            input,
            output,
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &GptqWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        weight.validate(self.input, self.output)?;
        self.operation.execute(
            &self.stream,
            &mut GptqLaunch {
                input,
                weight: i32_buffer(&weight.weight)?,
                zero_points: i32_buffer(&weight.zero_points)?,
                scales: weight.scales.as_f16().ok_or_else(|| Error::DTypeMismatch {
                    name: weight.scales.name().into(),
                    expected: "F16",
                })?,
                group_indices: i32_buffer(&weight.group_indices)?,
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
