//! Causal depthwise convolution, seeded with `kernel - 1` tokens of history.

use models::weights::TensorBinding;

use super::{MambaShape, StateSpaceState, dense_bias};
use crate::engine::{Array, Error, ModelTensors, Result, Stream, gated_delta::StateArray};

#[derive(Debug)]
pub(super) struct CausalConvolution {
    weight: Array,
    bias: Option<Array>,
    shape: MambaShape,
}

impl CausalConvolution {
    pub(super) fn load(
        tensors: &ModelTensors,
        binding: &TensorBinding,
        shape: MambaShape,
        stream: &Stream,
    ) -> Result<Self> {
        let weight = tensors.get(&binding.source)?;
        let dims = weight.shape()?;
        let channels = i32::try_from(shape.channels)?;
        let kernel = i32::try_from(shape.kernel)?;
        let weight = if dims == [channels, kernel, 1] {
            weight
        } else if dims == [channels, 1, kernel] {
            weight.transpose(&[0, 2, 1], stream)?
        } else {
            return Err(Error::InvalidModel(format!(
                "Mamba-2 convolution has unexpected shape {dims:?}"
            )));
        };
        let bias = dense_bias(binding)?.map(|name| tensors.get(name)).transpose()?;
        Ok(Self { weight, bias, shape })
    }

    /// Convolves `[rows, sequence, channels]`, updating each row's history,
    /// and applies the `silu` activation.
    pub(super) fn forward(
        &self,
        input: &Array,
        states: &mut [&mut StateSpaceState],
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let [rows, sequence, _] = shape.as_slice() else {
            return Err(Error::InvalidModel("Mamba-2 convolution input must have rank 3".into()));
        };
        let (rows, sequence) = (usize::try_from(*rows)?, usize::try_from(*sequence)?);
        let keep = self.shape.kernel - 1;
        let history = self.history(states, keep, input, stream)?;
        let graph = stream.native().graph();
        let joined = graph.concatenate(&[history.native(), input.native()], 1)?;
        let tail = graph.slice(
            &joined,
            &[0, sequence, 0],
            &[rows, sequence + keep, self.shape.channels],
        )?;
        let histories = split_rows(Array::from_native(tail)?, rows, stream)?;
        for (state, history) in states.iter_mut().zip(histories) {
            state.history = Some(history);
        }
        let mut output = graph.conv1d(
            &joined,
            self.weight.native(),
            1,
            0,
            1,
            i32::try_from(self.shape.channels)?,
        )?;
        if let Some(bias) = &self.bias {
            output = graph.add(&output, bias.native())?;
        }
        Array::from_native(graph.silu(&output)?)
    }

    fn history(
        &self,
        states: &[&mut StateSpaceState],
        keep: usize,
        input: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let dtype = input.native().dtype()?;
        let graph = stream.native().graph();
        let zeros = || -> Result<Array> {
            let shape = mirtal::Shape::new([1, keep, self.shape.channels])?;
            Array::from_native(graph.full(&shape, 0.0, dtype)?)
        };
        let rows = states
            .iter()
            .map(|state| {
                state
                    .history
                    .as_ref()
                    .map_or_else(|| zeros().map(StateArray::from), StateArray::try_clone)
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = rows.iter().collect::<Vec<_>>();
        StateArray::join(&refs, stream)
    }
}

/// Splits `[rows, ..]` into per-row state arrays; a single row stays whole.
pub(super) fn split_rows(array: Array, rows: usize, stream: &Stream) -> Result<Vec<StateArray>> {
    if rows == 1 {
        return Ok(vec![array.into()]);
    }
    StateArray::split(array, rows, stream)
}
