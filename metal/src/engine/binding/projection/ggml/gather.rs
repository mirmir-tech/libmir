//! Routed execution over a GGML expert bank, following MLX `gather_mm`
//! indexing: `input [.., 1, columns]` whose batch dimensions are a prefix of
//! `indices` (or equal to them) gives `indices.shape + [1, rows]`.

use super::{GgmlLinear, tokens};
use crate::engine::{Array, Error, Result, Stream};

/// The most assignments the gathered GEMV serves: it reads each selected
/// matrix once per assignment. Wider expert-sorted batches (prefill) run the
/// grouped simdgroup-matrix kernel; wider unsorted ones dequantize the bank
/// once for MLX's `gather_mm`.
const GATHERED_MAX_ASSIGNMENTS: usize = 512;

impl GgmlLinear {
    pub(in crate::engine) fn gather(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let experts = self.experts.ok_or_else(|| {
            Error::InvalidQuantization("gathered GGML execution requires an expert bank".into())
        })?;
        if self.bias.is_some() {
            return Err(Error::InvalidQuantization("GGML expert banks carry no bias here".into()));
        }
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
            if !sorted || tokens(input)? != assignments {
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
