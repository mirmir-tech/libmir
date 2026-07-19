use super::{attention::VisionAttention, mlp::VisionMlp};
use crate::engine::{Array, ModelTensors, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct EncoderLayer {
    input_norm: NormWeight,
    attention: VisionAttention,
    post_attention_norm: NormWeight,
    pre_feedforward_norm: NormWeight,
    mlp: VisionMlp,
    post_feedforward_norm: NormWeight,
    eps: f32,
}

impl EncoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        tensors: &ModelTensors,
        prefix: &str,
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            input_norm: norm(tensors, prefix, "input_layernorm")?,
            attention: VisionAttention::load(
                tensors,
                &format!("{prefix}.self_attn"),
                query_heads,
                key_value_heads,
                head_dim,
                rope_theta,
                eps,
                stream,
            )?,
            post_attention_norm: norm(tensors, prefix, "post_attention_layernorm")?,
            pre_feedforward_norm: norm(tensors, prefix, "pre_feedforward_layernorm")?,
            mlp: VisionMlp::load(tensors, &format!("{prefix}.mlp"), stream)?,
            post_feedforward_norm: norm(tensors, prefix, "post_feedforward_layernorm")?,
            eps,
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        positions: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let attention_input = self.input_norm.apply(input, self.eps, stream)?;
        let attention = self.attention.forward(&attention_input, positions, stream)?;
        let attention = self.post_attention_norm.apply(&attention, self.eps, stream)?;
        let hidden = input.add(&attention, stream)?;

        let mlp_input = self.pre_feedforward_norm.apply(&hidden, self.eps, stream)?;
        let mlp = self.mlp.forward(&mlp_input, stream)?;
        let mlp = self.post_feedforward_norm.apply(&mlp, self.eps, stream)?;
        hidden.add(&mlp, stream)
    }
}

fn norm(tensors: &ModelTensors, prefix: &str, name: &str) -> Result<NormWeight> {
    NormWeight::load(tensors, &format!("{prefix}.{name}"))
}
