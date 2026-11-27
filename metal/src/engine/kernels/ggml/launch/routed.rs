//! Launches over expert banks: the gathered GEMV and the grouped product.

use super::{SIMDGROUPS, output_shape, type_template};
use crate::engine::{
    Array, Error, Result, Stream,
    kernels::{ggml::GgmlMatrix, template},
};

/// `input [rows, columns]`, `indices [assignments]` with `assignments =
/// rows * group`: one routed product per assignment, `[assignments, rows of
/// one expert]`.
pub(in crate::engine::kernels::ggml) fn gathered(
    kernel: &mirtal::MetalKernel<3, 1>,
    stream: &Stream,
    [input, weight, indices]: [&Array; 3],
    matrix: GgmlMatrix,
    experts: usize,
) -> Result<Array> {
    let rows = expert_rows(matrix, experts)?;
    let assignments = expert_indices(indices)?;
    let (_, inputs) = output_shape(input, matrix.columns, rows)?;
    // No assignment would make a zero `GROUP` and an empty grid.
    if inputs == 0 || assignments == 0 || !assignments.is_multiple_of(inputs) {
        return Err(Error::InvalidQuantization(format!(
            "GGML gather of {assignments} assignments over {inputs} inputs and {experts} experts"
        )));
    }
    let dtype = input.native().dtype()?;
    let threads = 32 * SIMDGROUPS;
    let [output] = kernel.dispatch(
        stream.native(),
        [input.native(), weight.native(), indices.native()],
        &[mirtal::OutputSpec::new(mirtal::Shape::new([assignments, rows])?, dtype)],
        &mirtal::Dispatch::new(
            [rows.div_ceil(SIMDGROUPS) * threads, assignments, 1],
            [threads, 1, 1],
        )
        .templates([
            mirtal::TemplateArg::dtype("T", dtype),
            type_template(matrix)?,
            template("INPUT", matrix.columns)?,
            template("OUTPUT", rows)?,
            template("EXPERTS", experts)?,
            template("GROUP", assignments / inputs)?,
        ]),
    )?;
    Array::from_native(output)
}

/// `input [assignments, columns]` with expert-sorted `indices
/// [assignments]`: `[assignments, rows per expert]`.
pub(in crate::engine::kernels::ggml) fn grouped(
    kernel: &mirtal::MetalKernel<3, 1>,
    stream: &Stream,
    [input, weight, indices]: [&Array; 3],
    matrix: GgmlMatrix,
    experts: usize,
) -> Result<Array> {
    let rows = expert_rows(matrix, experts)?;
    let assignments = expert_indices(indices)?;
    let (_, inputs) = output_shape(input, matrix.columns, rows)?;
    if inputs != assignments || assignments == 0 {
        return Err(Error::InvalidQuantization(format!(
            "GGML grouped product of {inputs} inputs for {assignments} sorted assignments"
        )));
    }
    let dtype = input.native().dtype()?;
    let [output] = kernel.dispatch(
        stream.native(),
        [input.native(), weight.native(), indices.native()],
        &[mirtal::OutputSpec::new(mirtal::Shape::new([assignments, rows])?, dtype)],
        // Two simdgroups of two 8-row tiles over 16 assignments.
        &mirtal::Dispatch::new([rows.div_ceil(32) * 64, assignments.div_ceil(16), 1], [64, 1, 1])
            .templates([
                mirtal::TemplateArg::dtype("T", dtype),
                type_template(matrix)?,
                template("INPUT", matrix.columns)?,
                template("OUTPUT", rows)?,
                template("EXPERTS", experts)?,
                template("ASSIGNMENTS", assignments)?,
            ]),
    )?;
    Array::from_native(output)
}

/// The rows of one expert of `matrix`, a bank of `experts` equal experts;
/// checked before anything divides by `experts`.
fn expert_rows(matrix: GgmlMatrix, experts: usize) -> Result<usize> {
    if experts == 0 || !matrix.rows.is_multiple_of(experts) {
        return Err(Error::InvalidQuantization(format!(
            "GGML bank of {} rows over {experts} experts",
            matrix.rows
        )));
    }
    Ok(matrix.rows / experts)
}

/// The assignments `indices` holds, which the kernels read as `uint`.
fn expert_indices(indices: &Array) -> Result<usize> {
    let dtype = indices.native().dtype()?;
    if dtype != mirtal::DType::Uint32 {
        return Err(Error::InvalidQuantization(format!(
            "GGML expert indices of {dtype:?}, not Uint32"
        )));
    }
    Ok(indices.native().len())
}
