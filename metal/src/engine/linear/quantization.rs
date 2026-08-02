use super::{Array, Error, Result};

pub(in crate::engine) fn infer_bits(
    weight: &Array,
    scales: &Array,
    group_size: i32,
) -> Result<i32> {
    let group_size = usize::try_from(group_size)?;
    let packed = last_dimension(weight)?;
    let groups = last_dimension(scales)?;
    let input = groups.checked_mul(group_size).ok_or(Error::ShapeOverflow)?;
    let packed_bits = packed.checked_mul(32).ok_or(Error::ShapeOverflow)?;
    if input == 0 || packed_bits % input != 0 {
        return Err(Error::InvalidQuantization(format!(
            "packed={packed}, groups={groups}, group_size={group_size}"
        )));
    }
    let bits = i32::try_from(packed_bits / input)?;
    if matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        Ok(bits)
    } else {
        Err(Error::InvalidQuantization(format!("unsupported bit width {bits}")))
    }
}

fn last_dimension(array: &Array) -> Result<usize> {
    let Some(dimension) = array.shape()?.last().copied() else {
        return Err(Error::InvalidQuantization("scalar quantized tensor".into()));
    };
    Ok(usize::try_from(dimension)?)
}
