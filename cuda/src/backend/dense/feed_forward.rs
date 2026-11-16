use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    CudaBackend, DenseFeedForwardWeights, DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights,
    DownProjection, GateUpBuffers, GateUpProjection,
    routed::RoutedFeedForward,
    sandwich::{SandwichNorms, scalar},
};
use crate::{
    Error, GatedActivation, Result, RmsNormBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

/// Everything after attention in a dense layer, for a fixed number of rows:
/// the attention residual, the normalized gated MLP and its residual, plus
/// the sandwich norms and layer scalar of Gemma-style layers. Routed layers
/// take their experts in place of the gated MLP.
pub(super) struct DenseFeedForward {
    post_attention_norm: RmsNormBf16,
    sandwich: Option<SandwichNorms>,
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
            post_attention_norm: RmsNormBf16::new(backend, rows, hidden, epsilon)?,
            sandwich: weights
                .sandwich
                .map(|_| SandwichNorms::new(backend, rows, hidden, epsilon))
                .transpose()?,
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
    /// caller leaves in `scratch.attention`.
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        scratch: &mut DenseScratch,
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
        self.mlp(weights, scratch)?;
        let (Some(norms), Some(sandwich)) = (&self.sandwich, sandwich) else {
            return self.hidden_ops.add(&self.stream, &scratch.residual, &scratch.mlp, output);
        };
        norms.feed_forward_output.execute(
            &scratch.mlp,
            sandwich.feed_forward_output_norm,
            &mut scratch.normalized,
        )?;
        self.hidden_ops.add(
            &self.stream,
            &scratch.residual,
            &scratch.normalized,
            &mut scratch.attention,
        )?;
        self.hidden_ops.multiply_scalar(
            &self.stream,
            &scratch.attention,
            scalar(sandwich.layer_scalar)?,
            output,
        )
    }

    /// Writes the MLP of the normalized residual into `scratch.mlp`.
    fn mlp(&mut self, weights: DenseSwiGluWeights<'_>, scratch: &mut DenseScratch) -> Result<()> {
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
                    &scratch.residual,
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
                self.post_attention_norm.execute(
                    &scratch.residual,
                    weights.post_attention_norm,
                    &mut scratch.normalized,
                )?;
                routed.execute(&scratch.normalized, routed_weights, &mut scratch.mlp)
            },
            _ => Err(Error::InvalidExecutionPlan("dense layer MLP plan and weights differ")),
        }
    }
}
