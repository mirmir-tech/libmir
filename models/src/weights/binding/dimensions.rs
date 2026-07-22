use super::TensorBinding;
use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

pub(super) fn affine_geometry(
    source: &str,
    logical_shape: Option<&[usize]>,
    weight: &TensorInfo,
    scales: &TensorInfo,
) -> Result<(Option<u8>, Option<usize>)> {
    let Some(input) = logical_shape.and_then(|shape| shape.last()).copied() else {
        return Ok((None, None));
    };
    let groups = scales
        .shape
        .last()
        .copied()
        .ok_or_else(|| invalid(source, "scale shape is empty"))?;
    if groups == 0 || !input.is_multiple_of(groups) {
        return Err(invalid(source, "scale shape does not divide the logical input dimension"));
    }
    let bits = packed_bits(weight, input);
    Ok((bits, Some(input / groups)))
}

pub(super) fn uniform_affine_group_size(bindings: &[TensorBinding]) -> Option<usize> {
    let mut sizes = bindings.iter().filter_map(|binding| match binding.storage {
        super::TensorStorage::AffineQuantized { group_size, .. } => group_size,
        super::TensorStorage::PackedInt8 { .. }
        | super::TensorStorage::Dense { .. }
        | super::TensorStorage::BlockQuantized { .. }
        | super::TensorStorage::Auxiliary { .. } => None,
    });
    let first = sizes.next()?;
    sizes.all(|size| size == first).then_some(first)
}

fn packed_bits(weight: &TensorInfo, input: usize) -> Option<u8> {
    if weight.dtype == "U8" {
        return Some(8);
    }
    if weight.dtype != "U32" {
        return None;
    }
    let physical = *weight.shape.last()?;
    if physical == 0 || !input.is_multiple_of(physical) {
        return None;
    }
    let per_word = input / physical;
    if per_word == 0 || !32_usize.is_multiple_of(per_word) {
        return None;
    }
    u8::try_from(32 / per_word).ok()
}

fn invalid(source: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid affine binding for {source}: {reason}"))
}
