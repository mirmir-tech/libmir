use models::weights::{BitsAndBytesStorageDType, TensorBinding, TensorStorage};

use crate::engine::{Array, Dtype, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(in crate::engine) struct BitsAndBytes4BitLinear {
    weight: Array,
    absmax: Array,
    quant_map: Array,
    nested_absmax: Array,
    nested_quant_map: Array,
    input: usize,
    output: usize,
    block_size: usize,
    nested_block_size: Option<usize>,
    nested_offset_bits: u32,
}

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
) -> Result<BitsAndBytes4BitLinear> {
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
        return Err(invalid(binding, "requires bitsandbytes 4-bit storage"));
    };
    let Some([output, input]) = binding.logical_shape.as_deref() else {
        return Err(invalid(binding, "requires a logical matrix"));
    };
    let (output, input) = (*output, *input);
    let weight = tensors.get(&binding.source)?;
    require_bytes(&weight, output * input / 2, binding, "weight")?;
    require_dtype(&weight, storage_dtype(format.storage_dtype), binding, "weight")?;
    let absmax_array = tensors.get(absmax)?;
    let quant_map_array = tensors.get(quant_map)?;
    require(&quant_map_array, Dtype::Float32, &[16], binding, "quant map")?;
    let blocks = (output * input).div_ceil(format.block_size);
    let (nested_absmax_array, nested_quant_map_array) =
        if let Some(nested) = format.nested_block_size {
            require(&absmax_array, Dtype::Uint8, &[blocks], binding, "nested codes")?;
            let nested_absmax = tensors.get(
                nested_absmax
                    .as_deref()
                    .ok_or_else(|| invalid(binding, "nested absmax is missing"))?,
            )?;
            let nested_map = tensors.get(
                nested_quant_map
                    .as_deref()
                    .ok_or_else(|| invalid(binding, "nested map is missing"))?,
            )?;
            require(
                &nested_absmax,
                Dtype::Float32,
                &[blocks.div_ceil(nested)],
                binding,
                "nested absmax",
            )?;
            require(&nested_map, Dtype::Float32, &[256], binding, "nested map")?;
            (nested_absmax, nested_map)
        } else {
            require(&absmax_array, Dtype::Float32, &[blocks], binding, "absmax")?;
            (tensors.get(quant_map)?, tensors.get(quant_map)?)
        };
    Ok(BitsAndBytes4BitLinear {
        weight,
        absmax: absmax_array,
        quant_map: quant_map_array,
        nested_absmax: nested_absmax_array,
        nested_quant_map: nested_quant_map_array,
        input,
        output,
        block_size: format.block_size,
        nested_block_size: format.nested_block_size,
        nested_offset_bits: nested_offset_bits.unwrap_or(0),
    })
}

impl BitsAndBytes4BitLinear {
    pub(in crate::engine) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        if !matches!(input.dtype()?, Dtype::Float16 | Dtype::Bfloat16) {
            return Err(Error::InvalidQuantization(
                "bitsandbytes input must be F16 or BF16".into(),
            ));
        }
        stream.kernels().bitsandbytes_4bit_linear(
            [
                input,
                &self.weight,
                &self.absmax,
                &self.quant_map,
                &self.nested_absmax,
                &self.nested_quant_map,
            ],
            self.input,
            self.output,
            self.block_size,
            self.nested_block_size,
            self.nested_offset_bits,
            stream,
        )
    }
}

fn storage_dtype(dtype: BitsAndBytesStorageDType) -> Dtype {
    match dtype {
        BitsAndBytesStorageDType::U8 => Dtype::Uint8,
        BitsAndBytesStorageDType::F16 => Dtype::Float16,
        BitsAndBytesStorageDType::Bf16 => Dtype::Bfloat16,
        BitsAndBytesStorageDType::F32 => Dtype::Float32,
    }
}

fn require(
    array: &Array,
    dtype: Dtype,
    shape: &[usize],
    binding: &TensorBinding,
    kind: &str,
) -> Result<()> {
    let expected = shape
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if array.dtype()? == dtype && array.shape()? == expected {
        Ok(())
    } else {
        Err(invalid(binding, &format!("{kind} dtype or shape differs")))
    }
}

fn require_dtype(array: &Array, dtype: Dtype, binding: &TensorBinding, kind: &str) -> Result<()> {
    if array.dtype()? == dtype {
        Ok(())
    } else {
        Err(invalid(binding, &format!("{kind} dtype differs")))
    }
}

fn require_bytes(array: &Array, bytes: usize, binding: &TensorBinding, kind: &str) -> Result<()> {
    if array.byte_len()? == bytes {
        Ok(())
    } else {
        Err(invalid(binding, &format!("{kind} byte size differs")))
    }
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}
