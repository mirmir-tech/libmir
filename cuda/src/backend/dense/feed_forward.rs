use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    CudaBackend, DenseFeedForwardWeights, DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights,
    DownProjection, FeedForwardInput, GateUpBuffers, GateUpProjection,
    per_layer::PerLayerGate,
    routed::RoutedFeedForward,
    sandwich::{SandwichNorms, scalar},
};
use crate::{
    Error, GatedActivation, Result, RmsNormBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

/// Everything after attention in a dense layer, for a fixed number of rows:
/// the attention residual, the normalized gated MLP and its residual, plus
/// the sandwich norms and layer scalar of Gemma-style layers, and the
/// per-layer input gate of Gemma 4's E models. Routed layers take their
/// experts in place of the gated MLP.
pub(super) struct DenseFeedForward {
    post_attention_norm: RmsNormBf16,
    input: FeedForwardInput,
    sandwich: Option<SandwichNorms>,
    per_layer: Option<PerLayerGate>,
    mlp: Mlp,
    hidden_ops: ElementwiseBf16,
    stream: Stream,
}

#[expect(
    clippy::large_enum_variant,
    reason = "one per layer executor or template; the gated MLP is the common variant"
)]
enum Mlp {
    Gated {
        gate_up: GateUpProjection,
        down: DownProjection,
        activation: PackedGatedBf16,
        kind: GatedActivation,
    },
    Routed(Box<RoutedFeedForward>),
}

impl DenseFeedForward {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        rows: usize,
        weights: DenseSwiGluWeights<'_>,
    ) -> Result<Self> {
        let hidden = config.attention.hidden_size;
        let epsilon = config.attention.rms_norm_epsilon;
        Ok(Self {
            post_attention_norm: RmsNormBf16::for_stream(backend, rows, &config.attention)?,
            input: config.feed_forward_input,
            sandwich: weights
                .sandwich
                .map(|_| SandwichNorms::new(backend, rows, hidden, epsilon))
                .transpose()?,
            per_layer: match (config.per_layer_input, weights.per_layer_input) {
                (Some(width), Some(weights)) => Some(PerLayerGate::new(
                    backend,
                    rows,
                    [hidden, width.get().into()],
                    epsilon,
                    weights,
                )?),
                (None, None) => None,
                _ => {
                    return Err(Error::InvalidDecoderKernel(
                        "dense layer per-layer input mismatch",
                    ));
                },
            },
            mlp: match weights.feed_forward {
                DenseFeedForwardWeights::Gated { gate_up, down } => Mlp::Gated {
                    gate_up: GateUpProjection::new(backend, config, rows, Some(gate_up))?,
                    down: DownProjection::new(backend, config, rows, Some(down))?,
                    activation: PackedGatedBf16::compile(
                        &backend.inner.compiler,
                        rows,
                        config.intermediate_size,
                    )?,
                    kind: config.activation,
                },
                DenseFeedForwardWeights::Routed(weights) => {
                    Mlp::Routed(Box::new(RoutedFeedForward::new(backend, config, rows, weights)?))
                },
            },
            hidden_ops: ElementwiseBf16::compile(
                &backend.inner.compiler,
                super::scratch::elements(rows, hidden)?,
            )?,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Completes the layer from `input` and its attention output, which the
    /// caller leaves in `scratch.attention`; `per_layer` is the layer's
    /// per-layer input when it has one.
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        scratch: &mut DenseScratch,
        per_layer: Option<&DeviceBuffer<bf16>>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let sandwich = match (self.sandwich.is_some(), weights.sandwich) {
            (true, Some(sandwich)) => Some(sandwich),
            (false, None) => None,
            _ => return Err(Error::InvalidDecoderKernel("dense layer sandwich norms mismatch")),
        };
        if let (Some(norms), Some(sandwich)) = (&self.sandwich, sandwich) {
            norms.attention_output.execute(
                &scratch.attention,
                sandwich.attention_output_norm,
                &mut scratch.normalized,
            )?;
            self.hidden_ops
                .add(&self.stream, input, &scratch.normalized, &mut scratch.residual)?;
        } else {
            self.hidden_ops
                .add(&self.stream, input, &scratch.attention, &mut scratch.residual)?;
        }
        self.mlp(input, weights, scratch)?;
        if let (Some(norms), Some(sandwich)) = (&self.sandwich, sandwich) {
            norms.feed_forward_output.execute(
                &scratch.mlp,
                sandwich.feed_forward_output_norm,
                &mut scratch.normalized,
            )?;
        }
        let branch = if sandwich.is_some() {
            &scratch.normalized
        } else {
            &scratch.mlp
        };
        let layer_scalar = sandwich.and_then(|sandwich| sandwich.layer_scalar);
        let gate = match (&mut self.per_layer, weights.per_layer_input, per_layer) {
            (Some(gate), Some(weights), Some(input)) => Some((gate, weights, input)),
            (None, None, None) => None,
            _ => return Err(Error::InvalidDecoderKernel("dense layer per-layer input mismatch")),
        };
        if gate.is_none() && layer_scalar.is_none() {
            return self.hidden_ops.add(&self.stream, &scratch.residual, branch, output);
        }
        self.hidden_ops
            .add(&self.stream, &scratch.residual, branch, &mut scratch.attention)?;
        // `h` is in `scratch.attention`; the gate adds its per-layer input.
        let finished = if let Some((gate, weights, input)) = gate {
            let (hidden, normalized) = (&scratch.attention, &mut scratch.normalized);
            gate.execute(hidden, weights, input, &mut scratch.mlp, normalized)?;
            if layer_scalar.is_none() {
                return self.hidden_ops.add(&self.stream, hidden, &scratch.normalized, output);
            }
            let gated = &mut scratch.residual;
            self.hidden_ops.add(&self.stream, hidden, &scratch.normalized, gated)?;
            &scratch.residual
        } else {
            &scratch.attention
        };
        let layer_scalar =
            layer_scalar.ok_or(Error::InvalidDecoderKernel("dense layer lacks its scalar"))?;
        self.hidden_ops
            .multiply_scalar(&self.stream, finished, scalar(layer_scalar)?, output)
    }

    /// Writes the MLP of the normalized residual, or of the normalized layer
    /// input in parallel blocks, into `scratch.mlp`.
    fn mlp(
        &mut self,
        layer_input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        scratch: &mut DenseScratch,
    ) -> Result<()> {
        let source = match self.input {
            FeedForwardInput::AttentionResidual => &scratch.residual,
            FeedForwardInput::LayerInput => layer_input,
        };
        match (&mut self.mlp, weights.feed_forward) {
            (
                Mlp::Gated { gate_up, down, activation, kind },
                DenseFeedForwardWeights::Gated {
                    gate_up: gate_up_weights,
                    down: down_weights,
                },
            ) => {
                let separate = gate_up.execute(
                    &self.stream,
                    source,
                    &self.post_attention_norm,
                    weights.post_attention_norm,
                    gate_up_weights,
                    &mut GateUpBuffers {
                        normalized: &mut scratch.normalized,
                        packed: &mut scratch.gate_up,
                        separate: &mut scratch.gate_up_separate,
                    },
                )?;
                let fused_down = separate
                    && down.execute_gated(
                        &scratch.gate_up_separate[0],
                        &scratch.gate_up_separate[1],
                        *kind,
                        down_weights,
                        &mut scratch.mlp,
                    )?;
                if fused_down {
                    return Ok(());
                }
                if separate {
                    activation.execute_separate(
                        &self.stream,
                        &scratch.gate_up_separate[0],
                        &scratch.gate_up_separate[1],
                        &mut scratch.activated,
                        (*kind).into(),
                    )?;
                } else {
                    activation.execute(
                        &self.stream,
                        &scratch.gate_up,
                        &mut scratch.activated,
                        (*kind).into(),
                    )?;
                }
                down.execute(&self.stream, &scratch.activated, down_weights, &mut scratch.mlp)
            },
            (Mlp::Routed(routed), DenseFeedForwardWeights::Routed(routed_weights)) => {
                self.post_attention_norm.apply(
                    source,
                    weights.post_attention_norm,
                    &mut scratch.normalized,
                )?;
                routed.execute(&scratch.normalized, routed_weights, &mut scratch.mlp)
            },
            _ => Err(Error::InvalidExecutionPlan("dense layer MLP plan and weights differ")),
        }
    }
}
