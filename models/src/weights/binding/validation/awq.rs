use super::{companion, companion_dtype, divided, invalid, shape};
use crate::{
    error::Result,
    weights::{AwqQuantization, TensorBinding, TensorCatalog},
};

pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    format: AwqQuantization,
    scales: &str,
    zero_points: &str,
    catalog: &TensorCatalog,
) -> Result<()> {
    if !format.is_gemm_w4a16() {
        return Err(invalid(&binding.source, "AWQ format is outside the GEMM W4A16 contract"));
    }
    let [output, input] = logical else {
        return Err(invalid(&binding.source, "AWQ logical tensor is not a matrix"));
    };
    let packed_output = divided(*output, 8, &binding.source)?;
    shape(&binding.source, &binding.shape, &[*input, packed_output])?;
    let groups = divided(*input, format.group_size, &binding.source)?;
    companion(catalog, scales, &[groups, *output])?;
    companion_dtype(catalog, scales, "F16")?;
    companion(catalog, zero_points, &[groups, packed_output])?;
    companion_dtype(catalog, zero_points, "I32")
}
