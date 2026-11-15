//! One pre-norm transformer block: layer norm, fused QKV with bias,
//! bidirectional attention over the prefix and every patch, output projection,
//! residual; layer norm, exact GELU MLP, residual. No rotary positions: the
//! patches carry absolute positions from the embedding.

use super::{dimension, encoder};
use crate::engine::{Array, DenseLinear, LayerNorm, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct EncoderBlock {
    attention_norm: LayerNorm,
    qkv: DenseLinear,
    output: DenseLinear,
    mlp_norm: LayerNorm,
    up: DenseLinear,
    down: DenseLinear,
    heads: i32,
    head_dim: i32,
    scale: f32,
}

impl EncoderBlock {
    pub(super) fn load(
        tensors: &ModelTensors,
        index: usize,
        config: &models::layout::PixelShuffleVisionConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let name = |suffix: &str| encoder(&format!("blocks.{index}.{suffix}"));
        let eps: f32 = config.layer_norm_eps.to_string().parse()?;
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            attention_norm: LayerNorm::load(tensors, &name("norm1"), eps)?,
            qkv: DenseLinear::load(tensors, &name("attn.qkv"), stream)?,
            output: DenseLinear::load(tensors, &name("attn.proj"), stream)?,
            mlp_norm: LayerNorm::load(tensors, &name("norm2"), eps)?,
            up: DenseLinear::load(tensors, &name("mlp.fc1"), stream)?,
            down: DenseLinear::load(tensors, &name("mlp.fc2"), stream)?,
            heads: dimension(config.num_attention_heads, "attention heads")?,
            head_dim: dimension(head_dim, "attention head width")?,
            scale: head_dim.to_string().parse::<f32>()?.sqrt().recip(),
        })
    }

    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let attention = self.attention(&self.attention_norm.forward(input, stream)?, stream)?;
        let hidden = input.add(&attention, stream)?;
        let mlp = self.down.forward(
            &self
                .up
                .forward(&self.mlp_norm.forward(&hidden, stream)?, stream)?
                .gelu(stream)?,
            stream,
        )?;
        hidden.add(&mlp, stream)
    }

    fn attention(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let sequence = input.shape()?[1];
        let qkv = self
            .qkv
            .forward(input, stream)?
            .reshape(&[1, sequence, 3, self.heads, self.head_dim], stream)?
            .transpose(&[2, 0, 3, 1, 4], stream)?;
        let part = |index: usize| -> Result<Array> {
            let stop = [index + 1, 1, usize::try_from(self.heads)?, usize::try_from(sequence)?];
            let stop = [stop[0], stop[1], stop[2], stop[3], usize::try_from(self.head_dim)?];
            qkv.slice(&[index, 0, 0, 0, 0], &stop, stream)?
                .reshape(&[1, self.heads, sequence, self.head_dim], stream)
        };
        let attended = part(0)?.scaled_dot_product_attention(
            &part(1)?,
            &part(2)?,
            self.scale,
            false,
            stream,
        )?;
        self.output.forward(
            &attended
                .transpose(&[0, 2, 1, 3], stream)?
                .reshape(&[1, sequence, self.heads * self.head_dim], stream)?,
            stream,
        )
    }
}
