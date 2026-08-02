use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use super::{companion, companion_dtype, invalid, shape};
use crate::{
    error::Result,
    weights::{BitsAndBytes4BitType, TensorBinding, TensorCatalog, TensorStorage},
};

#[cfg(test)]
mod tests;

const NF4: [u32; 16] = [
    0xbf80_0000, 0xbf32_39b1, 0xbf06_6b30, 0xbeca_32a0, 0xbe91_a24d, 0xbe3d_353f, 0xbdba_7871,
    0x0000_0000, 0x3da2_faff, 0x3e24_cae3, 0x3e7c_04dd, 0x3ead_033a, 0x3ee1_a4b8, 0x3f10_07ab,
    0x3f39_13b3, 0x3f80_0000,
];
const FP4: [u32; 16] = [
    0x0000_0000, 0x3baa_aaab, 0x3f2a_aaab, 0x3f80_0000, 0x3eaa_aaab, 0x3f00_0000, 0x3e2a_aaab,
    0x3e80_0000, 0x0000_0000, 0xbbaa_aaab, 0xbf2a_aaab, 0xbf80_0000, 0xbeaa_aaab, 0xbf00_0000,
    0xbe2a_aaab, 0xbe80_0000,
];

pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    catalog: &TensorCatalog,
) -> Result<()> {
    let TensorStorage::BitsAndBytes4Bit {
        format,
        absmax,
        quant_map,
        nested_absmax,
        nested_quant_map,
        quant_state,
        nested_offset_bits,
    } = &binding.storage
    else {
        unreachable!()
    };
    if logical.len() != 2 || !format.is_supported() {
        return Err(invalid(&binding.source, "unsupported bitsandbytes matrix contract"));
    }
    let elements = logical
        .iter()
        .try_fold(1_usize, |n, value| n.checked_mul(*value))
        .ok_or_else(|| invalid(&binding.source, "logical element count overflows"))?;
    let weight = catalog
        .get(&binding.source)
        .ok_or_else(|| invalid(&binding.source, "weight tensor is missing"))?;
    if weight.payload_bytes()? != elements.div_ceil(2) {
        return Err(invalid(&binding.source, "packed 4-bit payload size differs"));
    }
    companion_dtype(catalog, &binding.source, format.storage_dtype.safetensors_name())?;
    let blocks = elements.div_ceil(format.block_size);
    companion(catalog, absmax, &[blocks])?;
    companion(catalog, quant_map, &[16])?;
    companion_dtype(catalog, quant_map, "F32")?;
    validate_codebook(catalog, quant_map, format.quant_type)?;
    companion_dtype(catalog, quant_state, "U8")?;
    if let Some(nested_block) = format.nested_block_size {
        let nested_absmax = nested_absmax
            .as_deref()
            .ok_or_else(|| invalid(&binding.source, "nested absmax is missing"))?;
        let nested_quant_map = nested_quant_map
            .as_deref()
            .ok_or_else(|| invalid(&binding.source, "nested quant map is missing"))?;
        if nested_offset_bits.is_none() {
            return Err(invalid(&binding.source, "nested offset is missing"));
        }
        companion_dtype(catalog, absmax, "U8")?;
        companion(catalog, nested_absmax, &[blocks.div_ceil(nested_block)])?;
        companion_dtype(catalog, nested_absmax, "F32")?;
        companion(catalog, nested_quant_map, &[256])?;
        companion_dtype(catalog, nested_quant_map, "F32")?;
    } else if nested_absmax.is_some() || nested_quant_map.is_some() || nested_offset_bits.is_some()
    {
        return Err(invalid(&binding.source, "unexpected nested quantization state"));
    } else {
        companion_dtype(catalog, absmax, "F32")?;
    }
    let state = catalog
        .get(quant_state)
        .ok_or_else(|| invalid(quant_state, "quant state is missing"))?;
    shape(quant_state, &state.shape, &[state.payload_bytes()?])
}

fn validate_codebook(
    catalog: &TensorCatalog,
    name: &str,
    kind: BitsAndBytes4BitType,
) -> Result<()> {
    let tensor = catalog.get(name).ok_or_else(|| invalid(name, "quant map is missing"))?;
    let mut file = File::open(&tensor.file)?;
    file.seek(SeekFrom::Start(tensor.payload_start()?))?;
    let mut bytes = vec![0; tensor.payload_bytes()?];
    file.read_exact(&mut bytes)?;
    let expected = match kind {
        BitsAndBytes4BitType::Nf4 => &NF4,
        BitsAndBytes4BitType::Fp4 => &FP4,
    };
    let (values, remainder) = bytes.as_chunks::<4>();
    let valid = remainder.is_empty()
        && values
            .iter()
            .zip(expected)
            .all(|(bytes, expected)| u32::from_le_bytes(*bytes) == *expected);
    if valid {
        Ok(())
    } else {
        Err(invalid(name, "bitsandbytes codebook differs"))
    }
}
