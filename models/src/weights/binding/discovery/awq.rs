use std::collections::BTreeSet;

use super::super::{
    AwqBits, AwqPacking, AwqQuantization, AwqScaleDType, AwqStorageDType, TensorStorage,
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
    let logical = logical.ok_or_else(|| invalid(&tensor.name, "logical shape is unavailable"))?;
    let [output, input] = logical else {
        return Err(invalid(&tensor.name, "logical tensor is not a matrix"));
    };
    let scales = format!("{prefix}.scales");
    let zero_points = format!("{prefix}.qzeros");
    let scale = catalog
        .get(&scales)
        .ok_or_else(|| invalid(&tensor.name, "scales are missing"))?;
    let zeros = catalog
        .get(&zero_points)
        .ok_or_else(|| invalid(&tensor.name, "qzeros are missing"))?;
    if tensor.dtype != "I32" || zeros.dtype != "I32" {
        return Err(invalid(&tensor.name, "qweight and qzeros must use I32"));
    }
    let packed_output = divided(*output, 8, &tensor.name)?;
    if tensor.shape != [*input, packed_output] {
        return Err(invalid(&tensor.name, "qweight is not input-major output-packed"));
    }
    let [groups, scale_output] = scale.shape.as_slice() else {
        return Err(invalid(&scale.name, "scales are not a matrix"));
    };
    if *groups == 0 || *scale_output != *output || !input.is_multiple_of(*groups) {
        return Err(invalid(&scale.name, "scale geometry does not match the logical matrix"));
    }
    if zeros.shape != [*groups, packed_output] {
        return Err(invalid(&zeros.name, "qzeros geometry does not match scales"));
    }
    let _inserted = consumed.insert(scales.clone());
    let _inserted = consumed.insert(zero_points.clone());
    Ok(TensorStorage::Awq {
        format: AwqQuantization {
            bits: AwqBits::Four,
            group_size: input / groups,
            packing: AwqPacking::GemmOutputInterleaved,
            storage_dtype: AwqStorageDType::I32,
            scale_dtype: AwqScaleDType::parse(scale)?,
            packed_zero_points: true,
        },
        scales,
        zero_points,
    })
}

fn divided(value: usize, divisor: usize, name: &str) -> Result<usize> {
    value
        .checked_div(divisor)
        .filter(|_| value > 0 && value.is_multiple_of(divisor))
        .ok_or_else(|| invalid(name, "dimension is not pack aligned"))
}

fn invalid(name: &str, reason: &str) -> ModelsError {
    ModelsError::InvalidConfig(format!("invalid AWQ binding {name}: {reason}"))
}
