use super::{companion, companion_dtype, divided, invalid, shape};
use crate::{
    error::Result,
    weights::{GptqQuantization, TensorBinding, TensorCatalog},
};

pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    format: GptqQuantization,
    scales: &str,
    zero_points: &str,
    group_indices: &str,
    catalog: &TensorCatalog,
) -> Result<()> {
    if !format.is_input_packed() {
        return Err(invalid(&binding.source, "GPTQ format is not input-packed I32"));
    }
    let [output, input] = logical else {
        return Err(invalid(&binding.source, "GPTQ logical tensor is not a matrix"));
    };
    let bits = usize::from(format.bits.get());
    let packed_input = divided(
        input
            .checked_mul(bits)
            .ok_or_else(|| invalid(&binding.source, "packed input width overflow"))?,
        32,
        &binding.source,
    )?;
    let packed_output = divided(
        output
            .checked_mul(bits)
            .ok_or_else(|| invalid(&binding.source, "packed output width overflow"))?,
        32,
        &binding.source,
    )?;
    let groups = divided(*input, format.group_size, &binding.source)?;
    shape(&binding.source, &binding.shape, &[packed_input, *output])?;
    companion(catalog, scales, &[groups, *output])?;
    companion_dtype(catalog, scales, format.scale_dtype.as_str())?;
    companion(catalog, zero_points, &[groups, packed_output])?;
    companion_dtype(catalog, zero_points, "I32")?;
    companion(catalog, group_indices, &[*input])?;
    companion_dtype(catalog, group_indices, "I32")
}
