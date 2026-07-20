use super::attention::EncoderAttention;
use crate::engine::{Array, DenseLinear, LayerNorm, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct EncoderLayer {
    attention: EncoderAttention,
    attention_norm: LayerNorm,
    up_gate: DenseLinear,
    down: DenseLinear,
    mlp_norm: LayerNorm,
    intermediate: usize,
}

impl EncoderLayer {
    pub fn load(
        tensors: &ModelTensors,
        index: usize,
        config: &models::layout::EncoderConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let prefix = format!("new.encoder.layer.{index}");
        let eps = config.layer_norm_eps.to_string().parse()?;
        Ok(Self {
            attention: EncoderAttention::load(
                tensors,
                &format!("{prefix}.attention"),
                config,
                stream,
            )?,
            attention_norm: LayerNorm::load(tensors, &format!("{prefix}.attn_ln"), eps)?,
            up_gate: DenseLinear::load(tensors, &format!("{prefix}.mlp.up_gate_proj"), stream)?,
            down: DenseLinear::load(tensors, &format!("{prefix}.mlp.down_proj"), stream)?,
            mlp_norm: LayerNorm::load(tensors, &format!("{prefix}.mlp_ln"), eps)?,
            intermediate: config.intermediate_size,
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let hidden = input.add(&self.attention.forward(input, stream)?, stream)?;
        let hidden = self.attention_norm.forward(&hidden, stream)?;
        let projected = self.up_gate.forward(&hidden, stream)?;
        let shape = projected.shape()?;
        let rows = usize::try_from(shape[1])?;
        let up = projected.slice(&[0, 0, 0], &[1, rows, self.intermediate], stream)?;
        let gate = projected.slice(
            &[0, 0, self.intermediate],
            &[1, rows, self.intermediate * 2],
            stream,
        )?;
        let activated = gate.gelu(stream)?.multiply(&up, stream)?;
        self.mlp_norm
            .forward(&hidden.add(&self.down.forward(&activated, stream)?, stream)?, stream)
    }
}
