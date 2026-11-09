//! Mamba-2 selective state-space mixer.

mod convolution;
mod recurrence;
mod state;

use models::{
    layout::{Mamba2Config, StateSpaceStateDType, TimeStepBound},
    weights::{StateSpaceMixerBindings, TensorBinding, TensorStorage},
};
use recurrence::RecurrenceWeights;
pub use state::StateSpaceState;

use crate::engine::{Array, Error, ModelTensors, Result, Stream, binding::BoundLinear};

/// Shapes fixed by the mixer configuration.
#[derive(Debug, Clone, Copy)]
pub(super) struct MambaShape {
    pub(super) heads: usize,
    pub(super) head_dim: usize,
    pub(super) state_size: usize,
    pub(super) groups: usize,
    pub(super) inner: usize,
    pub(super) channels: usize,
    pub(super) kernel: usize,
    pub(super) chunk: usize,
    pub(super) minimum_time_step: f32,
    pub(super) maximum_time_step: Option<f32>,
    pub(super) state_dtype: mirtal::DType,
}

#[derive(Debug)]
pub(super) struct Mamba2Mixer {
    input_projection: BoundLinear,
    convolution: convolution::CausalConvolution,
    recurrence: RecurrenceWeights,
    gated_norm: Array,
    output_projection: BoundLinear,
    epsilon: f32,
    shape: MambaShape,
}

impl Mamba2Mixer {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: StateSpaceMixerBindings<'_>,
        config: &Mamba2Config,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let shape = MambaShape {
            heads: config.heads,
            head_dim: config.head_dim,
            state_size: config.state_size,
            groups: config.groups,
            inner: config.inner_size(),
            channels: config.convolution_channels(),
            kernel: config.convolution_kernel_size,
            chunk: config.chunk_size,
            minimum_time_step: config.time_step_limit.minimum.to_string().parse()?,
            maximum_time_step: match config.time_step_limit.maximum {
                TimeStepBound::Finite(value) => Some(value.to_string().parse()?),
                TimeStepBound::Unbounded => None,
            },
            state_dtype: match config.state_dtype {
                StateSpaceStateDType::Float32 => mirtal::DType::Float32,
                StateSpaceStateDType::BFloat16 => mirtal::DType::Bfloat16,
                StateSpaceStateDType::Float16 => mirtal::DType::Float16,
            },
        };
        if !shape.state_size.is_multiple_of(32) || shape.state_size == 0 {
            return Err(Error::InvalidModel(
                "Mamba-2 decode requires a state size divisible by 32".into(),
            ));
        }
        Ok(Self {
            input_projection: BoundLinear::load(tensors, bindings.input_projection, stream)?,
            convolution: convolution::CausalConvolution::load(
                tensors,
                bindings.convolution,
                shape,
                stream,
            )?,
            recurrence: RecurrenceWeights {
                decay_log: tensors.get(&bindings.decay_log.source)?,
                skip_scale: tensors.get(&bindings.skip_scale.source)?,
                time_bias: tensors.get(&bindings.time_bias.source)?,
            },
            gated_norm: tensors.get(&bindings.gated_norm.source)?,
            output_projection: BoundLinear::load(tensors, bindings.output_projection, stream)?,
            epsilon,
            shape,
        })
    }

    /// Runs the mixer over `[rows, sequence, hidden]`, advancing every row
    /// state in `states`.
    pub(super) fn forward(
        &self,
        input: &Array,
        states: &mut [&mut StateSpaceState],
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let [rows, sequence, _] = shape.as_slice() else {
            return Err(Error::InvalidModel("Mamba-2 input must have rank three".into()));
        };
        let (rows, sequence) = (usize::try_from(*rows)?, usize::try_from(*sequence)?);
        if rows != states.len() {
            return Err(Error::InvalidModel("Mamba-2 row and state counts differ".into()));
        }
        let projected = self.input_projection.forward(input, stream)?;
        let inner = self.shape.inner;
        let (gate, conv_input, dt) = split_three(
            &projected,
            inner,
            self.shape.channels,
            self.shape.heads,
            rows,
            sequence,
            stream,
        )?;
        let conv = self.convolution.forward(&conv_input, states, stream)?;
        let group_width = self.shape.groups * self.shape.state_size;
        let (x, b, c) =
            split_three(&conv, inner, group_width, group_width, rows, sequence, stream)?;
        let y = recurrence::forward(&self.recurrence, self.shape, &x, &b, &c, &dt, states, stream)?;
        for state in states.iter_mut() {
            state.offset += sequence;
        }
        let gated = gate.silu_mul(&y, stream)?;
        let groups = i32::try_from(self.shape.groups)?;
        let group_size = i32::try_from(self.shape.inner / self.shape.groups)?;
        let grouped = gated.reshape(
            &[i32::try_from(rows)?, i32::try_from(sequence)?, groups, group_size],
            stream,
        )?;
        let normalized = grouped.rms_norm_unit(self.epsilon, stream)?;
        let flat = normalized.reshape(
            &[i32::try_from(rows)?, i32::try_from(sequence)?, i32::try_from(inner)?],
            stream,
        )?;
        let scaled = flat.multiply(&self.gated_norm, stream)?.astype_like(&flat, stream)?;
        self.output_projection.forward(&scaled, stream)
    }
}

/// Splits the last axis of `[rows, sequence, a + b + c]` into three arrays.
fn split_three(
    input: &Array,
    a: usize,
    b: usize,
    c: usize,
    rows: usize,
    sequence: usize,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let part = |start: usize, width: usize| {
        input.slice(&[0, 0, start], &[rows, sequence, start + width], stream)
    };
    Ok((part(0, a)?, part(a, b)?, part(a + b, c)?))
}

pub(super) fn dense_bias(binding: &TensorBinding) -> Result<Option<&str>> {
    match &binding.storage {
        TensorStorage::Dense { bias, .. } => Ok(bias.as_deref()),
        _ => Err(Error::InvalidModel("Mamba-2 convolution must have dense storage".into())),
    }
}
