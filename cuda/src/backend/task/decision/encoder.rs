use mircuda::{DeviceBuffer, f16};
use models::{
    decision::{DecisionTensor, EncoderTensor},
    layout::{ModernBertAttention, ModernBertConfig},
};

use super::{
    attention::{BLOCK, Band, attend_band, attend_full},
    batch::DeviceBatch,
    device::Device,
    plans::Input,
    rope,
    weights::{DeviceLinear, DeviceNorm, Uploader, Write},
};
use crate::{Result, kernels::RopeTables};

/// A `ModernBERT` encoder whose matrix products take `I` inputs; norms,
/// softmax statistics and the residual stream stay f32.
pub struct Encoder<I: Input> {
    embeddings: DeviceBuffer<f16>,
    embedding_norm: DeviceNorm,
    layers: Vec<Layer<I>>,
    final_norm: DeviceNorm,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    intermediate: usize,
    global: RopeTables,
    local: RopeTables,
}

struct Layer<I: Input> {
    /// `None` on layer 0, which attends to the embedding norm output.
    attention_norm: Option<DeviceNorm>,
    qkv: DeviceLinear<I>,
    output: DeviceLinear<I>,
    mlp_norm: DeviceNorm,
    mlp_input: DeviceLinear<I>,
    mlp_output: DeviceLinear<I>,
    attention: ModernBertAttention,
}

impl<I: Input> Encoder<I> {
    pub fn load(
        config: &ModernBertConfig,
        uploader: &Uploader<'_>,
        positions: usize,
    ) -> Result<Self> {
        let epsilon: f32 = config.norm_eps.to_string().parse()?;
        let layers = config
            .layers
            .iter()
            .enumerate()
            .map(|(layer, &attention)| {
                let role = |tensor| DecisionTensor::Encoder { layer, tensor };
                Ok(Layer {
                    attention_norm: (layer > 0)
                        .then(|| uploader.norm(role(EncoderTensor::AttentionNorm), None, epsilon))
                        .transpose()?,
                    qkv: uploader.linear(role(EncoderTensor::Qkv), None)?,
                    output: uploader.linear(role(EncoderTensor::AttentionOutput), None)?,
                    mlp_norm: uploader.norm(role(EncoderTensor::MlpNorm), None, epsilon)?,
                    mlp_input: uploader.linear(role(EncoderTensor::MlpInput), None)?,
                    mlp_output: uploader.linear(role(EncoderTensor::MlpOutput), None)?,
                    attention,
                })
            })
            .collect::<Result<_>>()?;
        let positions = positions.div_ceil(BLOCK) * BLOCK;
        let table = |theta: f64| {
            rope::tables(uploader.backend, theta.to_string().parse()?, positions, config.head_dim)
        };
        Ok(Self {
            embeddings: uploader.half(DecisionTensor::TokenEmbedding)?,
            embedding_norm: uploader.norm(DecisionTensor::EmbeddingNorm, None, epsilon)?,
            layers,
            final_norm: uploader.norm(DecisionTensor::FinalNorm, None, epsilon)?,
            heads: config.num_attention_heads,
            head_dim: config.head_dim,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            global: table(config.global_rope_theta)?,
            local: table(config.local_rope_theta)?,
        })
    }

    /// Widest band of a local layer.
    pub fn band(&self) -> Option<usize> {
        self.layers
            .iter()
            .filter_map(|layer| match layer.attention {
                ModernBertAttention::Local { radius } => Some(radius),
                ModernBertAttention::Global => None,
            })
            .max()
    }

    /// Final hidden states `[rows × length, hidden]`.
    pub fn forward(&self, device: &Device<'_>, batch: &DeviceBatch) -> Result<DeviceBuffer<f32>> {
        let stream = device.stream();
        let mut embedded = device.buffer(batch.tokens() * self.hidden)?;
        device
            .elementwise
            .embed(stream, &batch.tokens, &self.embeddings, &mut embedded)?;
        let mut hidden = device.norm::<f32>(&embedded, &self.embedding_norm)?;
        for layer in &self.layers {
            let input = match &layer.attention_norm {
                Some(norm) => device.norm::<I>(&hidden, norm)?,
                None => device.norm::<I>(&embedded, &self.embedding_norm)?,
            };
            let mixed = self.attend(device, layer, &input, batch)?;
            layer.output.apply(device, &mixed, &mut hidden, Write::Accumulate)?;
            let normed = device.norm::<I>(&hidden, &layer.mlp_norm)?;
            let expanded = device.linear(&layer.mlp_input, &normed)?;
            let mut activated = device.buffer::<I>(batch.tokens() * self.intermediate)?;
            I::geglu(device.layout, stream, &expanded, &mut activated, self.intermediate)?;
            layer.mlp_output.apply(device, &activated, &mut hidden, Write::Accumulate)?;
        }
        device.norm::<f32>(&hidden, &self.final_norm)
    }

    fn attend(
        &self,
        device: &Device<'_>,
        layer: &Layer<I>,
        input: &DeviceBuffer<I>,
        batch: &DeviceBatch,
    ) -> Result<DeviceBuffer<I>> {
        let qkv = device.linear(&layer.qkv, input)?;
        let shape = (batch.length, self.heads, self.head_dim);
        match layer.attention {
            ModernBertAttention::Global => {
                let mut rotated = device.buffer::<I>(qkv.len())?;
                I::rotate(device.layout, device.stream(), &qkv, &self.global, shape, &mut rotated)?;
                attend_full(device, (&rotated, &batch.lengths), shape)
            },
            ModernBertAttention::Local { radius } => {
                let band = Band {
                    length: batch.length,
                    heads: self.heads,
                    head_dim: self.head_dim,
                    radius,
                };
                attend_band(device, (&qkv, &batch.lengths), &self.local, band)
            },
        }
    }
}
