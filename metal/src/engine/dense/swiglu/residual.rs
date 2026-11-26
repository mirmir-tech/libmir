//! Residual joins and gate activation shared by the scalar and packed paths.

use super::layer::DenseSwiGluLayer;
use crate::engine::{Array, Result, Stream};

impl DenseSwiGluLayer {
    /// `input + branch`, the branch scaled by the residual multiplier when
    /// the decoder declares one (Granite).
    fn join(&self, input: &Array, branch: &Array, stream: &Stream) -> Result<Array> {
        match self.config.residual_multiplier {
            Some(multiplier) => input.add(&branch.multiply_scalar(multiplier, stream)?, stream),
            None => input.add(branch, stream),
        }
    }

    /// The stream after attention: `input + attention`, or in a parallel
    /// block (Cohere) `input + feed_forward(norm(input)) + attention`, the
    /// feed-forward then passing its input through.
    pub(super) fn attention_join(
        &self,
        input: &Array,
        attention: &Array,
        stream: &Stream,
        feed_forward: impl FnOnce(&Array) -> Result<Array>,
    ) -> Result<Array> {
        if self.weights.parallel() {
            feed_forward(input)?.add(attention, stream)
        } else {
            self.attention_residual(input, attention, stream)
        }
    }

    /// `input + attention`, with Gemma's attention-output norm when present.
    pub(super) fn attention_residual(
        &self,
        input: &Array,
        attention: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        match self.weights.output_norms() {
            Some(sandwich) => self.join(
                input,
                &sandwich.attention_output.apply(attention, self.config.rms_norm_eps, stream)?,
                stream,
            ),
            None => self.join(input, attention, stream),
        }
    }

    /// `input + feed_forward`, with Gemma's MLP-output norm when present.
    pub(super) fn feed_forward_residual(
        &self,
        input: &Array,
        feed_forward: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        match self.weights.output_norms() {
            Some(sandwich) => self.join(
                input,
                &sandwich.feed_forward_output.apply(
                    feed_forward,
                    self.config.rms_norm_eps,
                    stream,
                )?,
                stream,
            ),
            None => self.join(input, feed_forward, stream),
        }
    }

    /// `act(gate) * up` for this layer's gate activation.
    pub(super) fn gated(&self, gate: &Array, up: &Array, stream: &Stream) -> Result<Array> {
        self.config.activation.gate(gate, up, stream)
    }
}
