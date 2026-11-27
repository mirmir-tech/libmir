//! GGML types whose 32-value sub-blocks share one scale and minimum, repacked
//! once at load into MLX's affine groups of 32 so projections run MLX's
//! quantized products, whose cost over a few rows stays near one row's.

use models::gguf::{Affine32, GgmlType};

use super::{super::Kernels, GgmlMatrix, launch::type_template};
use crate::engine::{Array, Error, Result, Stream, kernels::template};

/// MLX's affine form of a GGML type ([`GgmlType::affine32`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::engine) struct AffineLayout(Affine32);

impl AffineLayout {
    /// MLX's affine groups of a GGML type.
    pub(in crate::engine) const GROUP_SIZE: i32 = 32;

    #[must_use]
    pub(in crate::engine) const fn of(ggml_type: GgmlType) -> Option<Self> {
        match ggml_type.affine32() {
            Some(layout) if layout.repacks_on_metal() => Some(Self(layout)),
            _ => None,
        }
    }

    #[must_use]
    pub(in crate::engine) const fn bits(self) -> i32 {
        self.0.bits as i32
    }
}

impl Kernels {
    /// `matrix` as MLX affine `[rows, columns * bits / 32]` words with
    /// `[rows, columns / 32]` scales and biases of `dtype`.
    pub(crate) fn ggml_affine(
        &self,
        stream: &Stream,
        weight: &Array,
        matrix: GgmlMatrix,
        layout: AffineLayout,
        dtype: mirtal::DType,
    ) -> Result<[Array; 3]> {
        if matrix.ggml_type.affine32() != Some(layout.0) || !matrix.columns.is_multiple_of(32) {
            return Err(Error::InvalidQuantization(format!(
                "{} has no affine layout {layout:?}",
                matrix.ggml_type
            )));
        }
        let (rows, groups) = (matrix.rows, matrix.columns / 32);
        let bits = usize::from(layout.0.bits);
        let outputs = [
            mirtal::OutputSpec::new(
                mirtal::Shape::new([rows, groups * bits])?,
                mirtal::DType::Uint32,
            ),
            mirtal::OutputSpec::new(mirtal::Shape::new([rows, groups])?, dtype),
            mirtal::OutputSpec::new(mirtal::Shape::new([rows, groups])?, dtype),
        ];
        let arrays = self.ggml.affine_repack.dispatch(
            stream.native(),
            [weight.native()],
            &outputs,
            &mirtal::Dispatch::new([groups, rows, 1], [groups.min(64), 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", dtype),
                type_template(matrix)?,
                template("INPUT", matrix.columns)?,
                template("BITS", bits)?,
                template("OFFSET", usize::from(layout.0.offset))?,
            ]),
        )?;
        arrays
            .map(Array::from_native)
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::InvalidQuantization("affine repack returned no arrays".into()))
    }
}

#[cfg(test)]
mod tests;
