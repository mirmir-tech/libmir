use super::{AffineBits, AffineStorageDType, TensorBinding};
use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

pub(super) fn transposes_last_two(logical: &[usize], physical: &[usize]) -> bool {
    if logical.len() < 2 || logical.len() != physical.len() || logical == physical {
        return false;
    }
    let mut transposed = logical.to_vec();
    let last = transposed.len() - 1;
    transposed.swap(last - 1, last);
    physical == transposed
}

pub(super) fn affine_geometry(
    source: &str,
    logical_shape: Option<&[usize]>,
    weight: &TensorInfo,
    scales: &TensorInfo,
) -> Result<(AffineBits, usize)> {
    let Some(input) = logical_shape.and_then(|shape| shape.last()).copied() else {
        return Err(invalid(source, "logical input dimension is unavailable"));
    };
    let groups = scales
        .shape
        .last()
        .copied()
        .ok_or_else(|| invalid(source, "scale shape is empty"))?;
    if groups == 0 || !input.is_multiple_of(groups) {
        return Err(invalid(source, "scale shape does not divide the logical input dimension"));
    }
    let bits = packed_bits(weight, input)?;
    Ok((bits, input / groups))
}

pub(super) fn uniform_affine_group_size(bindings: &[TensorBinding]) -> Option<usize> {
    let mut sizes = bindings.iter().filter_map(|binding| match binding.storage {
        super::TensorStorage::AffineQuantized { format, .. } => Some(format.group_size),
        super::TensorStorage::PackedInt8 { .. }
        | super::TensorStorage::PackedInt4 { .. }
        | super::TensorStorage::Awq { .. }
        | super::TensorStorage::Gptq { .. }
        | super::TensorStorage::BitsAndBytes4Bit { .. }
        | super::TensorStorage::Float8 { .. }
        | super::TensorStorage::Dense { .. }
        | super::TensorStorage::BlockQuantized { .. }
        | super::TensorStorage::Auxiliary { .. } => None,
    });
    let first = sizes.next()?;
    sizes.all(|size| size == first).then_some(first)
}

fn packed_bits(weight: &TensorInfo, input: usize) -> Result<AffineBits> {
    let storage = AffineStorageDType::parse(weight)?;
    let physical = weight
        .shape
        .last()
        .copied()
        .ok_or_else(|| invalid(&weight.name, "packed shape is empty"))?;
    let packed_bits = physical
        .checked_mul(storage.bits())
        .ok_or_else(|| invalid(&weight.name, "packed bit count overflows"))?;
    let bits = packed_bits
        .checked_div(input)
        .filter(|bits| input > 0 && packed_bits.is_multiple_of(input) && *bits > 0)
        .ok_or_else(|| {
            invalid(&weight.name, "packed shape does not encode an integral bit width")
        })?;
    AffineBits::try_from(
        u8::try_from(bits).map_err(|_| invalid(&weight.name, "packed bit width exceeds u8"))?,
    )
}

fn invalid(source: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid affine binding for {source}: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::transposes_last_two;

    #[test]
    fn square_shapes_do_not_claim_an_ambiguous_transpose() {
        assert!(!transposes_last_two(&[896, 896], &[896, 896]));
        assert!(transposes_last_two(&[128, 896], &[896, 128]));
    }
}
