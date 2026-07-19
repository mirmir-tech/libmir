use super::{attention::VisionAttention, mlp::VisionMlp};
use crate::engine::{Array, LayerNorm, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct EncoderLayer {
    attention_norm: LayerNorm,
    attention: VisionAttention,
    mlp_norm: LayerNorm,
    mlp: VisionMlp,
}

impl EncoderLayer {
    pub(super) fn load(
        tensors: &ModelTensors,
        prefix: &str,
        heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            attention_norm: LayerNorm::load(tensors, &format!("{prefix}.norm1"), 1.0e-6)?,
            attention: VisionAttention::load(
                tensors,
                &format!("{prefix}.attn"),
                heads,
                head_dim,
                stream,
            )?,
            mlp_norm: LayerNorm::load(tensors, &format!("{prefix}.norm2"), 1.0e-6)?,
            mlp: VisionMlp::load(tensors, &format!("{prefix}.mlp"), stream)?,
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        positions: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let attention = self.attention.forward(
            &self.attention_norm.forward(input, stream)?,
            positions,
            stream,
        )?;
        let hidden = input.add(&attention, stream)?;
        hidden.add(&self.mlp.forward(&self.mlp_norm.forward(&hidden, stream)?, stream)?, stream)
    }
}
