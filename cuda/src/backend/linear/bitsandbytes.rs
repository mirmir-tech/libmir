use mircuda::{DeviceBuffer, Stream, bf16};
use models::weights::{BitsAndBytes4BitQuantization, TensorBinding, TensorStorage};

use crate::{
    CudaBackend, CudaTensorSet, Error, Result,
    kernels::{BitsAndBytes4BitLaunch, BitsAndBytes4BitLinear, BitsAndBytes4BitSpec},
};

#[derive(Clone, Debug)]
pub struct BitsAndBytes4BitWeight {
    weight: DeviceBuffer<u8>,
    absmax: DeviceBuffer<u8>,
    quant_map: DeviceBuffer<u8>,
    nested_absmax: DeviceBuffer<u8>,
    nested_quant_map: DeviceBuffer<u8>,
    nested_offset: f32,
    format: BitsAndBytes4BitQuantization,
    input: usize,
    output: usize,
}

impl BitsAndBytes4BitWeight {
    pub(crate) fn load_binding(
        tensors: &CudaTensorSet,
        binding: &TensorBinding,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let TensorStorage::BitsAndBytes4Bit {
            format,
            absmax,
            quant_map,
            nested_absmax,
            nested_quant_map,
            nested_offset_bits,
            ..
        } = &binding.storage
        else {
            return Err(Error::InvalidQuantizedGemv("binding is not bitsandbytes 4-bit"));
        };
        let weight_tensor = required(tensors, &binding.source)?;
        let absmax_tensor = required(tensors, absmax)?;
        let map_tensor = required(tensors, quant_map)?;
        let nested_absmax_tensor =
            nested_absmax.as_deref().map(|name| required(tensors, name)).transpose()?;
        let nested_map_tensor =
            nested_quant_map.as_deref().map(|name| required(tensors, name)).transpose()?;
        let dummy = absmax_tensor.raw_u8()?;
        let value = Self {
            weight: weight_tensor.raw_u8()?,
            absmax: absmax_tensor.raw_u8()?,
            quant_map: map_tensor.raw_u8()?,
            nested_absmax: nested_absmax_tensor
                .map_or_else(|| Ok(dummy.clone()), crate::CudaTensor::raw_u8)?,
            nested_quant_map: nested_map_tensor
                .map_or_else(|| Ok(dummy), crate::CudaTensor::raw_u8)?,
            nested_offset: nested_offset_bits.map_or(0.0, f32::from_bits),
            format: *format,
            input,
            output,
        };
        value.validate()?;
        Ok(value)
    }

    pub(in crate::backend) fn validate(&self) -> Result<()> {
        let elements = self
            .input
            .checked_mul(self.output)
            .ok_or(Error::InvalidQuantizedGemv("bitsandbytes matrix size overflows"))?;
        require(self.weight.len(), elements.div_ceil(2), "weight")?;
        require(self.quant_map.len(), 16 * 4, "quant map")?;
        let blocks = elements.div_ceil(self.format.block_size);
        if let Some(nested) = self.format.nested_block_size {
            require(self.absmax.len(), blocks, "nested codes")?;
            require(self.nested_absmax.len(), blocks.div_ceil(nested) * 4, "nested absmax")?;
            require(self.nested_quant_map.len(), 256 * 4, "nested quant map")?;
        } else {
            require(self.absmax.len(), blocks * 4, "absmax")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BitsAndBytes4BitBf16Linear {
    operation: BitsAndBytes4BitLinear,
    stream: Stream,
}

impl BitsAndBytes4BitBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        weight: &BitsAndBytes4BitWeight,
    ) -> Result<Self> {
        weight.validate()?;
        let spec = BitsAndBytes4BitSpec::new(
            tokens,
            weight.input,
            weight.output,
            weight.format.block_size,
            weight.format.nested_block_size,
        )?;
        Ok(Self {
            operation: BitsAndBytes4BitLinear::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &BitsAndBytes4BitWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        weight.validate()?;
        self.operation.execute(
            &self.stream,
            &mut BitsAndBytes4BitLaunch {
                input,
                weight: &weight.weight,
                absmax: &weight.absmax,
                quant_map: &weight.quant_map,
                nested_absmax: &weight.nested_absmax,
                nested_quant_map: &weight.nested_quant_map,
                nested_offset: weight.nested_offset,
                output,
            },
        )
    }
}

fn required<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a crate::CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}

fn require(actual: usize, expected: usize, kind: &'static str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidQuantizedGemv(kind))
    }
}
