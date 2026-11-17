//! Residual joins and gate activation shared by the scalar and packed paths.

use super::{config::GateActivation, layer::DenseSwiGluLayer};
use crate::engine::{Array, Result, Stream};

impl DenseSwiGluLayer {
    /// `input + attention`, with Gemma's attention-output norm when present.
    pub(super) fn attention_residual(
        &self,
        input: &Array,
        attention: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.weights.sandwich {
            Some(sandwich) => input.add(
                &sandwich.attention_output.apply(attention, self.config.rms_norm_eps, stream)?,
                stream,
            ),
            None => input.add(attention, stream),
        }
    }

    /// `input + feed_forward`, with Gemma's MLP-output norm when present.
    pub(super) fn feed_forward_residual(
        &self,
        input: &Array,
        feed_forward: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.weights.sandwich {
            Some(sandwich) => input.add(
                &sandwich.feed_forward_output.apply(
                    feed_forward,
                    self.config.rms_norm_eps,
                    stream,
                )?,
                stream,
            ),
            None => input.add(feed_forward, stream),
        }
    }

    /// `act(gate) * up` for this layer's gate activation.
    pub(super) fn gated(&self, gate: &Array, up: &Array, stream: &Stream) -> Result<Array> {
        match self.config.activation {
            GateActivation::Silu => gate.silu_mul(up, stream),
            GateActivation::GeluTanh => gate.gelu_approx_mul(up, stream),
        }
    }
}
