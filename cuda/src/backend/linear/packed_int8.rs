use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{
    CompressedIntegerScaleDType, CompressedIntegerScaleStrategy, TensorBinding, TensorStorage,
};

use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    kernels::{
        PackedInt8Embedding, PackedInt8EmbeddingLaunch, PackedInt8EmbeddingSpec, PackedInt8Launch,
        PackedInt8Linear, PackedInt8Spec,
    },
};

#[derive(Clone, Debug)]
pub struct CompressedInt8Weight {
    weight: CudaTensor,
    scales: CudaTensor,
    bits: usize,
    group_size: usize,
}

impl CompressedInt8Weight {
    pub(crate) fn load_binding(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let (TensorStorage::PackedInt8 {
            format,
            scales,
            zero_points,
            group_indices,
            ..
        }
        | TensorStorage::PackedInt4 {
            format,
            scales,
            zero_points,
            group_indices,
            ..
        }) = &binding.storage
        else {
            return Err(Error::InvalidQuantizedGemv(
                "binding is not a compressed-tensors packed integer weight",
            ));
        };
        let group_size = match format.scale_strategy {
            CompressedIntegerScaleStrategy::Channel if format.is_symmetric_channel_int8() => input,
            CompressedIntegerScaleStrategy::Group { group_size }
                if format.is_symmetric_group_int4() =>
            {
                group_size
            },
            _ => {
                return Err(Error::InvalidQuantizedGemv(
                    "CUDA packed integer requires the symmetric INT8 or grouped INT4 contract",
                ));
            },
        };
        if format.scale_dtype != CompressedIntegerScaleDType::BF16
            || zero_points.is_some()
            || group_indices.is_some()
        {
            return Err(Error::InvalidQuantizedGemv(
                "CUDA packed integer requires BF16 scales without zero points or group indices",
            ));
        }
        let value = Self {
            weight: required(tensors, &binding.source)?,
            scales: required(tensors, scales)?,
            bits: usize::from(format.bits.get()),
            group_size,
        };
        value.validate(input, output)?;
        Ok(value)
    }

    pub(in crate::backend) fn validate(&self, input: usize, output: usize) -> Result<()> {
        shape(&self.weight, &[output, input * self.bits / 32])?;
        shape(&self.scales, &[output, input / self.group_size])?;
        dtype(&self.weight, CudaTensorDType::I32, "I32")?;
        dtype(&self.scales, CudaTensorDType::Bf16, "BF16")
    }
}

#[derive(Clone, Debug)]
pub(in crate::backend) struct CompressedInt8Embedding {
    operation: PackedInt8Embedding,
    stream: Stream,
    weight: CompressedInt8Weight,
    vocab: usize,
}

impl CompressedInt8Embedding {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        vocab: usize,
        hidden: usize,
        output_scale: f32,
        weight: CompressedInt8Weight,
    ) -> Result<Self> {
        weight.validate(hidden, vocab)?;
        Ok(Self {
            operation: PackedInt8Embedding::compile(
                &backend.inner.compiler,
                PackedInt8EmbeddingSpec::new_packed(
                    vocab,
                    hidden,
                    output_scale,
                    weight.bits,
                    weight.group_size,
                )?,
            )?,
            stream: backend.inner.stream.clone(),
            weight,
            vocab,
        })
    }

    pub(in crate::backend) fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
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
        self.operation.execute(
            &self.stream,
            &mut PackedInt8EmbeddingLaunch {
                selected,
                selected_start,
                tokens,
                weight,
                scales,
                output,
            },
        )
    }

    pub(in crate::backend) fn validate_token(&self, token: u32) -> Result<()> {
        if usize::try_from(token)? < self.vocab {
            Ok(())
        } else {
            Err(Error::InvalidToken { token, vocab: self.vocab })
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompressedInt8Bf16Linear {
    operation: PackedInt8Linear,
    stream: Stream,
    input: usize,
    output: usize,
}

impl CompressedInt8Bf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        weight: &CompressedInt8Weight,
    ) -> Result<Self> {
        weight.validate(input, output)?;
        Ok(Self {
            operation: PackedInt8Linear::compile(
                &backend.inner.compiler,
                PackedInt8Spec::new_packed(tokens, input, output, weight.bits, weight.group_size)?,
            )?,
            stream: backend.inner.stream.clone(),
            input,
            output,
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &CompressedInt8Weight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        weight.validate(self.input, self.output)?;
        let packed = weight.weight.as_i32().ok_or_else(|| Error::DTypeMismatch {
            name: weight.weight.name().into(),
            expected: "I32",
        })?;
        let scales = weight.scales.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.scales.name().into(),
            expected: "BF16",
        })?;
        self.operation
            .execute(&self.stream, &mut PackedInt8Launch { input, weight: packed, scales, output })
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
