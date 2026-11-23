//! Routed `SwiGLU` experts with no shared expert (`Qwen3-MoE`) in place of the
//! dense layer's gated MLP: softmax top-k routing renormalized over the
//! selection, then the gathered expert products and their weighted sum.

mod sparse_mixer;

use models::weights::RoutedFeedForwardBindings;

use super::projection::BoundLinear;
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

/// Routes up to this many token selections unsorted, each assignment
/// reading its expert directly; more (prefill) sort selections by expert.
const UNSORTED_MAX_ASSIGNMENTS: usize = 512;

#[derive(Debug)]
pub(super) struct RoutedMlp {
    router: BoundLinear,
    gate: BoundLinear,
    up: BoundLinear,
    down: BoundLinear,
    top_k: i32,
    /// Sparsemixer jitter; softmax top-k renormalized over the selection
    /// otherwise.
    sparse_mixer: Option<f32>,
}

impl RoutedMlp {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: RoutedFeedForwardBindings<'_>,
        top_k: usize,
        sparse_mixer: Option<f32>,
        stream: &Stream,
    ) -> Result<Self> {
        let models::weights::RoutedExpertBindings::SeparateGateUp { gate, up, down } =
            bindings.experts
        else {
            return Err(Error::InvalidModel(
                "dense Metal routed experts require stacked separate gate and up banks".into(),
            ));
        };
        Ok(Self {
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            gate: BoundLinear::load(tensors, gate, stream)?,
            up: BoundLinear::load(tensors, up, stream)?,
            down: BoundLinear::load(tensors, down, stream)?,
            top_k: i32::try_from(top_k)?,
            sparse_mixer,
        })
    }

    /// `input [batch, sequence, hidden]`, already normalized.
    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let routing = match self.sparse_mixer {
            None => self.router.route_unit(input, self.top_k, stream)?,
            Some(jitter) => {
                let logits = self.router.forward(input, stream)?;
                let graph = stream.native().graph();
                let routing =
                    sparse_mixer::route(&logits, usize::try_from(self.top_k)?, jitter, graph)?;
                // Expert reductions take weights in the activations' dtype.
                crate::engine::RouterOutput {
                    weights: routing.weights.astype_like(input, stream)?,
                    indices: routing.indices,
                }
            },
        };
        let assignments = routing.indices.native().len();
        if assignments <= UNSORTED_MAX_ASSIGNMENTS {
            let expanded = input.expand_dims(&[-2, -3], stream)?;
            return self
                .experts(&expanded, &routing.indices, false, stream)?
                .squeeze_axis(-2, stream)?
                .weighted_sum(&routing.weights, -2, stream);
        }
        let sorted = input.sort_expert_inputs(&routing.indices, stream)?;
        let output = self.experts(&sorted.input, &sorted.indices, true, stream)?;
        // One fused gather and weighted sum instead of the graph's restore and
        // strided reduction (Qwen3-30B Q4_K_M 2048-token prefill +3%).
        sorted.restore_weighted(&output, &routing.weights, stream)
    }

    fn experts(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let gate = self.gate.gather(input, indices, sorted, stream)?;
        let up = self.up.gather(input, indices, sorted, stream)?;
        self.down.gather(&gate.silu_mul(&up, stream)?, indices, sorted, stream)
    }
}
