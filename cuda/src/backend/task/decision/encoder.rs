use mircuda::{DeviceBuffer, f16};
use models::{
    decision::{DecisionTensor, EncoderTensor},
    layout::{ModernBertAttention, ModernBertConfig},
};

use super::{
    batch::DeviceBatch,
    weights::{DeviceLinear, DeviceNorm, Plans, Uploader, Write},
};
use crate::{
    CudaBackend, Result,
    kernels::{DecisionAttention, DecisionDenseAttention, DecisionElementwise, DecisionWindow},
};

/// A `ModernBERT` encoder with f32 weights resident on the device.
pub struct Encoder {
    embeddings: DeviceBuffer<f16>,
    embedding_norm: DeviceNorm,
    layers: Vec<Layer>,
    final_norm: DeviceNorm,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    intermediate: usize,
    global_theta: f32,
    local_theta: f32,
}

struct Layer {
    /// `None` on layer 0, which attends to the embedding norm output.
    attention_norm: Option<DeviceNorm>,
    qkv: DeviceLinear,
    output: DeviceLinear,
    mlp_norm: DeviceNorm,
    mlp_input: DeviceLinear,
    mlp_output: DeviceLinear,
    attention: ModernBertAttention,
}

/// Kernels and backend every forward step needs.
pub struct Device<'a> {
    pub backend: &'a CudaBackend,
    pub elementwise: &'a DecisionElementwise,
    pub attention: &'a DecisionAttention,
    pub dense: &'a DecisionDenseAttention,
    pub plans: &'a Plans,
}

impl Device<'_> {
    pub const fn kit(&self) -> (&CudaBackend, &DecisionElementwise, &Plans) {
        (self.backend, self.elementwise, self.plans)
    }

    pub fn buffer(&self, elements: usize) -> Result<DeviceBuffer<f32>> {
        Ok(self.backend.inner.pool.allocate(&self.backend.inner.stream, elements)?)
    }

    pub fn norm(&self, input: &DeviceBuffer<f32>, norm: &DeviceNorm) -> Result<DeviceBuffer<f32>> {
        let mut output = self.buffer(input.len())?;
        self.elementwise.norm(
            &self.backend.inner.stream,
            input,
            (&norm.weight, &norm.bias, norm.epsilon),
            &mut output,
        )?;
        Ok(output)
    }

    pub fn linear(
        &self,
        linear: &DeviceLinear,
        input: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>> {
        let mut output = self.buffer(input.len() / linear.inputs * linear.outputs)?;
        linear.apply(self.kit(), input, &mut output, Write::Overwrite)?;
        Ok(output)
    }
}

impl Encoder {
    pub fn load(config: &ModernBertConfig, uploader: &Uploader<'_>) -> Result<Self> {
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
        Ok(Self {
            embeddings: uploader.half(DecisionTensor::TokenEmbedding)?,
            embedding_norm: uploader.norm(DecisionTensor::EmbeddingNorm, None, epsilon)?,
            layers,
            final_norm: uploader.norm(DecisionTensor::FinalNorm, None, epsilon)?,
            heads: config.num_attention_heads,
            head_dim: config.head_dim,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            global_theta: config.global_rope_theta.to_string().parse()?,
            local_theta: config.local_rope_theta.to_string().parse()?,
        })
    }

    /// Final hidden states `[rows × length, hidden]`.
    pub fn forward(&self, device: &Device<'_>, batch: &DeviceBatch) -> Result<DeviceBuffer<f32>> {
        let stream = &device.backend.inner.stream;
        let mut embedded = device.buffer(batch.tokens() * self.hidden)?;
        device
            .elementwise
            .embed(stream, &batch.tokens, &self.embeddings, &mut embedded)?;
        let mut hidden = device.norm(&embedded, &self.embedding_norm)?;
        for layer in &self.layers {
            let mixed = match &layer.attention_norm {
                Some(norm) => self.attend(device, layer, &device.norm(&hidden, norm)?, batch)?,
                None => self.attend(device, layer, &hidden, batch)?,
            };
            layer.output.apply(device.kit(), &mixed, &mut hidden, Write::Accumulate)?;
            let expanded =
                device.linear(&layer.mlp_input, &device.norm(&hidden, &layer.mlp_norm)?)?;
            let mut activated = device.buffer(batch.tokens() * self.intermediate)?;
            device.elementwise.geglu(stream, &expanded, &mut activated, self.intermediate)?;
            layer
                .mlp_output
                .apply(device.kit(), &activated, &mut hidden, Write::Accumulate)?;
        }
        device.norm(&hidden, &self.final_norm)
    }

    fn attend(
        &self,
        device: &Device<'_>,
        layer: &Layer,
        input: &DeviceBuffer<f32>,
        batch: &DeviceBatch,
    ) -> Result<DeviceBuffer<f32>> {
        let stream = &device.backend.inner.stream;
        let mut qkv = device.linear(&layer.qkv, input)?;
        let (theta, window) = match layer.attention {
            ModernBertAttention::Global => (self.global_theta, DecisionWindow::Full),
            ModernBertAttention::Local { radius } => {
                (self.local_theta, DecisionWindow::Band { radius })
            },
        };
        let rope = (batch.length, 3 * self.hidden, 2 * self.heads, self.head_dim, theta);
        device.elementwise.rope(stream, &mut qkv, rope)?;
        let mut mixed = device.buffer(batch.tokens() * self.hidden)?;
        device.dense.execute(
            stream,
            (&qkv, &batch.lengths),
            (batch.length, self.heads, self.head_dim, window),
            &mut mixed,
        )?;
        Ok(mixed)
    }
}
