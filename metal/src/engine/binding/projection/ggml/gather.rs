//! Routed execution over a GGML expert bank, following MLX `gather_mm`
//! indexing: `input [.., 1, columns]` whose batch dimensions are a prefix of
//! `indices` (or equal to them) gives `indices.shape + [1, rows]`.

use super::{GgmlLinear, scaled, tokens};
use crate::engine::{Array, Error, Result, Stream};

/// The most assignments the gathered GEMV serves: it reads each selected
/// matrix once per assignment. Wider expert-sorted batches (prefill) run the
/// grouped simdgroup-matrix kernel; wider unsorted ones dequantize the bank
/// once for MLX's `gather_mm`.
const GATHERED_MAX_ASSIGNMENTS: usize = 512;

/// Sorted assignments per expert from which dequantizing the whole bank for
/// MLX's sorted `gather_mm` beats the grouped kernel: its cost follows the
/// bank, the grouped kernel's the assignments. On M5 Pro (Qwen3-30B shapes,
/// 128 experts) they cross near 40-48; at 128 per expert the bank path takes
/// 4.0-4.6 instead of 8.3-10.2 ms per projection.
const DENSE_MIN_ASSIGNMENTS_PER_EXPERT: usize = 48;

impl GgmlLinear {
    /// The routed products, each scaled by its expert's output scale.
    pub(in crate::engine) fn gather(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let super::GgmlLayout::Bank { experts, scale } = &self.layout else {
            return Err(Error::InvalidQuantization(
                "gathered GGML execution requires an expert bank".into(),
            ));
        };
        let output = self.gather_blocks(input, indices, *experts, sorted, stream)?;
        let Some(scale) = scale else {
            return Ok(output);
        };
        let mut shape = indices.shape()?;
        shape.extend([1, 1]);
        scaled(&output, &scale.take(indices, 0, stream)?.reshape(&shape, stream)?, stream)
    }

    fn gather_blocks(
        &self,
        input: &Array,
        indices: &Array,
        experts: usize,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let rows = self.matrix.rows / experts;
        let input_shape = input.shape()?;
        if input_shape.len() < 2 || input_shape[input_shape.len() - 2] != 1 {
            return Err(Error::InvalidQuantization(format!(
                "GGML gather input {input_shape:?} is not [.., 1, columns]"
            )));
        }
        let assignments = indices.native().len();
        let mut output_shape = indices.shape()?;
        output_shape.extend([1, i32::try_from(rows)?]);
        if assignments > GATHERED_MAX_ASSIGNMENTS {
            if !sorted
                || tokens(input)? != assignments
                || assignments >= DENSE_MIN_ASSIGNMENTS_PER_EXPERT * experts
            {
                return self.gather_dense(input, indices, experts, sorted, stream);
            }
            let flat_input = input.reshape(
                &[i32::try_from(assignments)?, i32::try_from(self.matrix.columns)?],
                stream,
            )?;
            let flat_indices = indices.reshape(&[i32::try_from(assignments)?], stream)?;
            return stream
                .kernels()
                .ggml_grouped(
                    stream,
                    [&flat_input, &self.weight, &flat_indices],
                    self.matrix,
                    experts,
                )?
                .reshape(&output_shape, stream);
        }
        let flat_input = input.reshape(
            &[i32::try_from(tokens(input)?)?, i32::try_from(self.matrix.columns)?],
            stream,
        )?;
        let flat_indices = indices.reshape(&[i32::try_from(assignments)?], stream)?;
        stream
            .kernels()
            .ggml_gathered(
                stream,
                [&flat_input, &self.weight, &flat_indices],
                self.matrix,
                experts,
            )?
            .reshape(&output_shape, stream)
    }

    /// The bank dequantized to the activation type, `[experts, columns, rows]`,
    /// multiplied by MLX `gather_mm`.
    fn gather_dense(
        &self,
        input: &Array,
        indices: &Array,
        experts: usize,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let rows = i32::try_from(self.matrix.rows / experts)?;
        let dense = stream.kernels().ggml_dequantize(
            stream,
            &self.weight,
            self.matrix,
            input.native().dtype()?,
        )?;
        let bank = dense
            .reshape(&[i32::try_from(experts)?, rows, i32::try_from(self.matrix.columns)?], stream)?
            .transpose(&[0, 2, 1], stream)?;
        Array::from_native(stream.native().graph().gather_mm(
            input.native(),
            bank.native(),
            mirtal::GatherIndices { lhs: None, rhs: Some(indices.native()) },
            sorted,
        )?)
    }
}
