use mircuda::{DeviceBuffer, bf16};
use models::weights::{TensorBinding, TensorStorage};

use super::{
    CompressedInt8Bf16Linear, CompressedInt8Embedding, CompressedInt8Weight,
    awq::{AwqBf16Linear, AwqWeight},
    bitsandbytes::{BitsAndBytes4BitBf16Linear, BitsAndBytes4BitWeight},
    gptq::{GptqBf16Linear, GptqWeight},
};
use crate::{CudaBackend, CudaTensorSet, Error, Result};

#[derive(Clone, Debug)]
pub enum PackedIntegerWeight {
    Compressed(CompressedInt8Weight),
    Awq(AwqWeight),
    Gptq(GptqWeight),
    BitsAndBytes4Bit(BitsAndBytes4BitWeight),
}

impl PackedIntegerWeight {
    pub(crate) fn load_binding(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        match binding.storage {
            TensorStorage::PackedInt8 { .. } | TensorStorage::PackedInt4 { .. } => {
                CompressedInt8Weight::load_binding(tensors, binding, input, output)
                    .map(Self::Compressed)
            },
            TensorStorage::Awq { .. } => {
                AwqWeight::load_binding(tensors, binding, input, output).map(Self::Awq)
            },
            TensorStorage::Gptq { .. } => {
                GptqWeight::load_binding(tensors, binding, input, output).map(Self::Gptq)
            },
            TensorStorage::BitsAndBytes4Bit { .. } => {
                BitsAndBytes4BitWeight::load_binding(tensors, binding, input, output)
                    .map(Self::BitsAndBytes4Bit)
            },
            _ => Err(Error::InvalidQuantizedGemv("binding is not a packed integer weight")),
        }
    }

    pub(in crate::backend) fn validate(&self, input: usize, output: usize) -> Result<()> {
        match self {
            Self::Compressed(weight) => weight.validate(input, output),
            Self::Awq(weight) => weight.validate(input, output),
            Self::Gptq(weight) => weight.validate(input, output),
            Self::BitsAndBytes4Bit(weight) => weight.validate(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PackedIntegerBf16Linear {
    Compressed(CompressedInt8Bf16Linear),
    Awq(AwqBf16Linear),
    Gptq(GptqBf16Linear),
    BitsAndBytes4Bit(BitsAndBytes4BitBf16Linear),
}

impl PackedIntegerBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        weight: &PackedIntegerWeight,
    ) -> Result<Self> {
        match weight {
            PackedIntegerWeight::Compressed(weight) => {
                CompressedInt8Bf16Linear::new(backend, tokens, input, output, weight)
                    .map(Self::Compressed)
            },
            PackedIntegerWeight::Awq(weight) => {
                AwqBf16Linear::new(backend, tokens, input, output, weight).map(Self::Awq)
            },
            PackedIntegerWeight::Gptq(weight) => {
                GptqBf16Linear::new(backend, tokens, input, output, weight).map(Self::Gptq)
            },
            PackedIntegerWeight::BitsAndBytes4Bit(weight) => {
                BitsAndBytes4BitBf16Linear::new(backend, tokens, weight).map(Self::BitsAndBytes4Bit)
            },
        }
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &PackedIntegerWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, weight) {
            (Self::Compressed(operation), PackedIntegerWeight::Compressed(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Awq(operation), PackedIntegerWeight::Awq(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::Gptq(operation), PackedIntegerWeight::Gptq(weight)) => {
                operation.execute(input, weight, output)
            },
            (Self::BitsAndBytes4Bit(operation), PackedIntegerWeight::BitsAndBytes4Bit(weight)) => {
                operation.execute(input, weight, output)
            },
            _ => Err(Error::InvalidExecutionPlan("packed integer operation/weight mismatch")),
        }
    }
}

pub(in crate::backend) fn embedding(
    backend: &CudaBackend,
    vocab: usize,
    hidden: usize,
    output_scale: f32,
    weight: PackedIntegerWeight,
) -> Result<CompressedInt8Embedding> {
    match weight {
        PackedIntegerWeight::Compressed(weight) => {
            CompressedInt8Embedding::new(backend, vocab, hidden, output_scale, weight)
        },
        PackedIntegerWeight::Awq(_)
        | PackedIntegerWeight::Gptq(_)
        | PackedIntegerWeight::BitsAndBytes4Bit(_) => Err(Error::InvalidQuantizedGemv(
            "AWQ/GPTQ storage cannot be used as an embedding table",
        )),
    }
}
