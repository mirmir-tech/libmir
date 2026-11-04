use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    CudaBackend, DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights, DownProjection,
    GateUpBuffers, GateUpProjection,
    sandwich::{SandwichNorms, scalar},
};
use crate::{
    Error, GatedActivation, Result, RmsNormBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

/// Everything after attention in a dense layer, for a fixed number of rows:
/// the attention residual, the normalized gated MLP and its residual, plus
/// the sandwich norms and layer scalar of Gemma-style layers.
pub(super) struct DenseFeedForward {
    post_attention_norm: RmsNormBf16,
    sandwich: Option<SandwichNorms>,
    gate_up: GateUpProjection,
    down: DownProjection,
    activation: PackedGatedBf16,
    hidden_ops: ElementwiseBf16,
    stream: Stream,
    kind: GatedActivation,
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
            gate_up: GateUpProjection::new(backend, config, rows, Some(weights.gate_up))?,
            down: DownProjection::new(backend, config, rows, Some(weights.down))?,
            activation: PackedGatedBf16::compile(
                &backend.inner.compiler,
                rows,
                config.intermediate_size,
            )?,
            hidden_ops: ElementwiseBf16::compile(
                &backend.inner.compiler,
                super::scratch::elements(rows, hidden)?,
            )?,
            stream: backend.inner.stream.clone(),
            kind: config.activation,
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

    /// Writes the gated MLP of the normalized residual into `scratch.mlp`.
    fn mlp(&mut self, weights: DenseSwiGluWeights<'_>, scratch: &mut DenseScratch) -> Result<()> {
        let separate = self.gate_up.execute(
            &self.stream,
            &scratch.residual,
            &self.post_attention_norm,
            weights.post_attention_norm,
            weights.gate_up,
            &mut GateUpBuffers {
                normalized: &mut scratch.normalized,
                packed: &mut scratch.gate_up,
                separate: &mut scratch.gate_up_separate,
            },
        )?;
        let fused_down = separate
            && self.down.execute_gated(
                &scratch.gate_up_separate[0],
                &scratch.gate_up_separate[1],
                self.kind,
                weights.down,
                &mut scratch.mlp,
            )?;
        if fused_down {
            return Ok(());
        }
        if separate {
            self.activation.execute_separate(
                &self.stream,
                &scratch.gate_up_separate[0],
                &scratch.gate_up_separate[1],
                &mut scratch.activated,
                self.kind.into(),
            )?;
        } else {
            self.activation.execute(
                &self.stream,
                &scratch.gate_up,
                &mut scratch.activated,
                self.kind.into(),
            )?;
        }
        self.down
            .execute(&self.stream, &scratch.activated, weights.down, &mut scratch.mlp)
    }
}
