use std::collections::BTreeSet;

use super::super::{
    CompressedIntegerActivationOrder, CompressedIntegerBits, CompressedIntegerPacking,
    CompressedIntegerQuantization, CompressedIntegerScaleDType, CompressedIntegerScaleStrategy,
    CompressedIntegerSignedness, CompressedIntegerStorageDType, CompressedIntegerZeroPointMode,
    TensorStorage,
};
use crate::{
    error::{ModelsError, Result},
    weights::{TensorCatalog, TensorInfo},
};

pub(super) fn storage(
    prefix: &str,
    logical: Option<&[usize]>,
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
) -> Result<TensorStorage> {
    let scales = format!("{prefix}.weight_scale");
    let shape = format!("{prefix}.weight_shape");
    let scale = catalog.get(&scales).ok_or_else(|| {
        ModelsError::InvalidConfig(format!("missing packed INT8 scales {scales}"))
    })?;
    let _shape_tensor = catalog.get(&shape).ok_or_else(|| {
        ModelsError::InvalidConfig(format!("missing packed INT8 logical shape {shape}"))
    })?;
    let logical = logical.ok_or_else(|| {
        ModelsError::InvalidConfig(format!(
            "packed INT8 tensor has no logical role: {}",
            tensor.name
        ))
    })?;
    let input = logical.last().copied().ok_or_else(|| {
        ModelsError::InvalidConfig(format!("packed INT8 tensor has empty shape: {}", tensor.name))
    })?;
    let packed = tensor.shape.last().copied().ok_or_else(|| {
        ModelsError::InvalidConfig(format!("packed INT8 tensor has empty storage: {}", tensor.name))
    })?;
    let packed_bits = packed
        .checked_mul(32)
        .ok_or_else(|| ModelsError::InvalidConfig("packed INT8 width overflow".into()))?;
    if input == 0 || !packed_bits.is_multiple_of(input) {
        return Err(ModelsError::InvalidConfig(format!(
            "packed INT8 width is not integral: {}",
            tensor.name
        )));
    }
    let bits = CompressedIntegerBits::try_from(u8::try_from(packed_bits / input)?)?;
    if tensor.dtype != "I32" {
        return Err(ModelsError::InvalidConfig(format!(
            "packed INT8 words must use I32: {}",
            tensor.name
        )));
    }
    let zero_points = companion(prefix, "weight_zero_point", catalog);
    let group_indices = companion(prefix, "weight_g_idx", catalog);
    if zero_points.is_some() || group_indices.is_some() {
        return Err(ModelsError::InvalidConfig(format!(
            "asymmetric or activation-ordered packed INT8 is not admitted: {}",
            tensor.name
        )));
    }
    let _inserted = consumed.insert(scales.clone());
    let _inserted = consumed.insert(shape.clone());
    let groups = scale.shape.last().copied().ok_or_else(|| {
        ModelsError::InvalidConfig(format!("packed integer scale is empty: {}", scale.name))
    })?;
    let scale_strategy = if groups == 1 {
        CompressedIntegerScaleStrategy::Channel
    } else if groups > 0 && input.is_multiple_of(groups) {
        CompressedIntegerScaleStrategy::Group { group_size: input / groups }
    } else {
        return Err(ModelsError::InvalidConfig(format!(
            "packed integer scales do not divide the input: {}",
            scale.name
        )));
    };
    let format = CompressedIntegerQuantization {
        bits,
        scale_strategy,
        signedness: CompressedIntegerSignedness::OffsetBinary,
        zero_point: CompressedIntegerZeroPointMode::None,
        activation_order: CompressedIntegerActivationOrder::None,
        packing: CompressedIntegerPacking::DenseLittleEndian,
        storage_dtype: CompressedIntegerStorageDType::I32,
        scale_dtype: CompressedIntegerScaleDType::parse(scale)?,
    };
    let storage = match bits {
        CompressedIntegerBits::Four => TensorStorage::PackedInt4 {
            format,
            scales,
            shape,
            zero_points,
            group_indices,
        },
        CompressedIntegerBits::Eight => TensorStorage::PackedInt8 {
            format,
            scales,
            shape,
            zero_points,
            group_indices,
        },
    };
    Ok(storage)
}

fn companion(prefix: &str, suffix: &str, catalog: &TensorCatalog) -> Option<String> {
    let name = format!("{prefix}.{suffix}");
    catalog.contains(&name).then_some(name)
}
