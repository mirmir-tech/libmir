//! Gemma 4's E-model layers: attending over an earlier layer's keys and
//! values, and gating a per-layer input into the output.

use std::sync::Arc;

use models::{semantic::DecoderLayerSpec, weights::SharedKeyValueLayerBindings};

use super::{
    super::{
        HybridMoeLayerConfig,
        sharing::{PerLayerGate, StepShare},
        weights::LayerWeights,
    },
    HybridMoeLayer,
};
use crate::engine::{Array, ModelTensors, Result, Stream};

impl HybridMoeLayer {
    /// A layer whose attention reads layer `source`'s keys and values of the
    /// same step.
    pub(in crate::engine::hybrid::moe) fn load_shared_dense_bindings(
        tensors: &ModelTensors,
        bindings: &SharedKeyValueLayerBindings<'_>,
        layer: &DecoderLayerSpec,
        (step, source): (Arc<StepShare>, usize),
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let config = config.validate()?;
        let frequencies = Array::rope_periods(super::super::attention(layer)?)?;
        let weights = LayerWeights::load_shared_dense_bindings(
            tensors,
            bindings,
            frequencies,
            (step, source),
            stream,
        )?;
        Self::finish_load(weights, config, stream)
    }

    #[must_use]
    pub(in crate::engine::hybrid::moe) fn with_per_layer_gate(
        mut self,
        gate: Option<PerLayerGate>,
    ) -> Self {
        self.per_layer_gate = gate;
        self
    }

    /// The feed-forward output with the per-layer input gated in, before the
    /// layer scalar.
    pub(in crate::engine::hybrid::moe) fn gate_per_layer_input(
        &self,
        hidden: Array,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.per_layer_gate {
            Some(gate) => gate.apply(&hidden, self.config.rms_norm_eps, stream),
            None => Ok(hidden),
        }
    }
}
