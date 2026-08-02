use super::{companion, companion_dtype, invalid, projection, shape};
use crate::{
    error::Result,
    weights::{
        AffineGroupAxis, AffinePacking, AffineSignedness, AffineZeroPointMode,
        GroupedAffineQuantization, TensorBinding, TensorCatalog,
    },
};

pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    format: GroupedAffineQuantization,
    scales: &str,
    biases: Option<&str>,
    catalog: &TensorCatalog,
) -> Result<()> {
    if format.group_axis != AffineGroupAxis::Input
        || format.signedness != AffineSignedness::Unsigned
        || format.packing != AffinePacking::Mlx
    {
        return Err(invalid(&binding.source, "storage is not native MLX grouped affine"));
    }
    let has_bias = biases.is_some();
    if has_bias != (format.zero_point == AffineZeroPointMode::AdditiveBias)
        || has_bias != format.bias_dtype.is_some()
    {
        return Err(invalid(&binding.source, "zero-point mode and bias companion disagree"));
    }
    if let Some(dtype) = format.bias_dtype
        && dtype != format.scale_dtype
    {
        return Err(invalid(&binding.source, "scale and bias dtypes differ"));
    }
    let (prefix, input) = projection(logical, &binding.source)?;
    let packed = input
        .checked_mul(usize::from(format.bits.get()))
        .filter(|value| value.is_multiple_of(format.storage_dtype.bits()))
        .map(|value| value / format.storage_dtype.bits())
        .ok_or_else(|| {
            invalid(&binding.source, "logical input cannot be packed into storage dtype")
        })?;
    let mut physical = prefix.to_vec();
    physical.push(packed);
    shape(&binding.source, &binding.shape, &physical)?;

    let groups = input
        .checked_div(format.group_size)
        .filter(|groups| input.is_multiple_of(format.group_size) && *groups > 0)
        .ok_or_else(|| invalid(&binding.source, "invalid affine group geometry"))?;
    let mut parameters = prefix.to_vec();
    parameters.push(groups);
    companion(catalog, scales, &parameters)?;
    companion_dtype(catalog, scales, format.scale_dtype.as_str())?;
    if let Some(biases) = biases {
        companion(catalog, biases, &parameters)?;
        let dtype = format.bias_dtype.ok_or_else(|| invalid(biases, "bias dtype is missing"))?;
        companion_dtype(catalog, biases, dtype.as_str())?;
    }

    let weight = catalog
        .get(&binding.source)
        .ok_or_else(|| invalid(&binding.source, "packed tensor is missing"))?;
    if weight.dtype != format.storage_dtype.as_str() {
        return Err(invalid(&binding.source, "packed dtype disagrees with affine format"));
    }
    Ok(())
}
