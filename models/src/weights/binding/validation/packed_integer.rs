use super::{companion, companion_dtype, divided, invalid, projection, shape};
use crate::{
    error::Result,
    weights::{CompressedIntegerScaleStrategy, TensorBinding, TensorCatalog, TensorStorage},
};

pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    storage: &TensorStorage,
    catalog: &TensorCatalog,
) -> Result<()> {
    let (TensorStorage::PackedInt8 {
        format,
        scales,
        shape: shape_tensor,
        zero_points,
        group_indices,
    }
    | TensorStorage::PackedInt4 {
        format,
        scales,
        shape: shape_tensor,
        zero_points,
        group_indices,
    }) = storage
    else {
        return Err(invalid(&binding.source, "storage kind changed during validation"));
    };
    if !(format.is_symmetric_channel_int8() || format.is_symmetric_group_int4())
        || zero_points.is_some()
        || group_indices.is_some()
    {
        return Err(invalid(
            &binding.source,
            "packed integer storage is outside the admitted symmetric contract",
        ));
    }
    let (prefix, input) = projection(logical, &binding.source)?;
    let packed_bits = input
        .checked_mul(usize::from(format.bits.get()))
        .ok_or_else(|| invalid(&binding.source, "packed width overflow"))?;
    let packed = divided(packed_bits, 32, &binding.source)?;
    let mut physical = prefix.to_vec();
    physical.push(packed);
    shape(&binding.source, &binding.shape, &physical)?;
    let mut scale_shape = prefix.to_vec();
    scale_shape.push(match format.scale_strategy {
        CompressedIntegerScaleStrategy::Channel => 1,
        CompressedIntegerScaleStrategy::Group { group_size } => {
            divided(input, group_size, &binding.source)?
        },
    });
    companion(catalog, scales, &scale_shape)?;
    companion_dtype(catalog, scales, format.scale_dtype.as_str())?;
    companion(catalog, shape_tensor, &[logical.len()])?;
    companion_dtype(catalog, shape_tensor, "I64")
}
