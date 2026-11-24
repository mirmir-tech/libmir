use models::layout::DecoderConfig;

use super::layer::DecoderLayerTemplate;
use crate::{
    CudaBackend, CudaTensor, DecodeMoeLayerTemplate, DenseSwiGluLayerTemplate, Error, PagedKvCache,
    Result,
    backend::{
        linear::CheckpointProjectionWeight,
        model::{
            boundary::{ModelEmbeddingTemplate, ModelOutputHeadTemplate},
            per_layer::PerLayerInputsTemplate,
        },
    },
};

mod instantiate;

/// Immutable CUDA model weights and layer templates shared across sessions.
pub struct CudaMoeModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: ModelEmbeddingTemplate,
    final_norm: CudaTensor,
    output_head: ModelOutputHeadTemplate,
    layers: Vec<DecoderLayerTemplate>,
    logit_shaping: Option<crate::kernels::LogitShaping>,
    per_layer_inputs: Option<PerLayerInputsTemplate>,
}

impl CudaMoeModelTemplate {
    pub(crate) fn new(
        backend: &CudaBackend,
        decoder: DecoderConfig,
        embedding_scale: f32,
        embedding: CheckpointProjectionWeight,
        final_norm: CudaTensor,
        output_head: CheckpointProjectionWeight,
        layers: Vec<DecodeMoeLayerTemplate>,
    ) -> Result<Self> {
        let layers = layers
            .into_iter()
            .map(|layer| DecoderLayerTemplate::Moe(Box::new(layer)))
            .collect();
        Self::new_layers(
            backend, decoder, embedding, final_norm, output_head, layers, embedding_scale,
        )
    }

    pub(crate) fn new_dense_bound(
        backend: &CudaBackend,
        decoder: DecoderConfig,
        embedding_scale: f32,
        embedding: CheckpointProjectionWeight,
        final_norm: CudaTensor,
        output_head: CheckpointProjectionWeight,
        layers: Vec<DenseSwiGluLayerTemplate>,
    ) -> Result<Self> {
        let layers = layers
            .into_iter()
            .map(|layer| DecoderLayerTemplate::Dense(Box::new(layer)))
            .collect();
        Self::new_layers(
            backend, decoder, embedding, final_norm, output_head, layers, embedding_scale,
        )
    }

    fn new_layers(
        backend: &CudaBackend,
        decoder: DecoderConfig,
        embedding: CheckpointProjectionWeight,
        final_norm: CudaTensor,
        output_head: CheckpointProjectionWeight,
        layers: Vec<DecoderLayerTemplate>,
        embedding_scale: f32,
    ) -> Result<Self> {
        validate(&decoder, &final_norm, &layers)?;
        let embedding = ModelEmbeddingTemplate::new(
            embedding,
            decoder.vocab_size,
            decoder.hidden_size,
            embedding_scale,
        )?;
        let output_head = ModelOutputHeadTemplate::prepare(
            backend,
            output_head,
            decoder.hidden_size,
            decoder.vocab_size,
        )?;
        let logit_shaping = crate::kernels::LogitShaping::new(
            decoder.logits_divisor,
            decoder.final_logit_softcapping,
        )?;
        Ok(Self {
            backend: backend.clone(),
            decoder,
            embedding,
            final_norm,
            output_head,
            layers,
            logit_shaping,
            per_layer_inputs: None,
        })
    }

    /// Gives every layer its per-layer input (Gemma 4's E models).
    pub(crate) fn with_per_layer_inputs(mut self, inputs: Option<PerLayerInputsTemplate>) -> Self {
        self.per_layer_inputs = inputs;
        self
    }

    #[must_use]
    pub const fn decoder(&self) -> &DecoderConfig {
        &self.decoder
    }

    /// One cache per layer; a layer reading an earlier layer's keys and
    /// values shares that layer's pages.
    pub(crate) fn allocate_shared_kv(&self) -> Result<Vec<PagedKvCache>> {
        let mut caches: Vec<PagedKvCache> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let attention = layer.attention();
            let cache = match attention.key_value_source {
                Some(source) => caches
                    .get(usize::from(source))
                    .cloned()
                    .ok_or(Error::InvalidDecoderKernel("shared keys come from a later layer"))?,
                None => self.backend.prepare_paged_kv(attention.layer, attention.cache)?,
            };
            caches.push(cache);
        }
        Ok(caches)
    }
}

fn validate(
    decoder: &DecoderConfig,
    final_norm: &CudaTensor,
    layers: &[DecoderLayerTemplate],
) -> Result<()> {
    // The final norm follows the first layer's formula (a biased LayerNorm
    // packs its bias after the weight).
    let norm = layers.first().map_or(decoder.hidden_size, |layer| {
        layer.attention().stream_norm.weight_len(decoder.hidden_size)
    });
    if final_norm.shape() != [norm]
        || layers.len() != decoder.num_hidden_layers
        || layers.iter().enumerate().any(|(index, layer)| {
            layer.attention().layer != index || layer.attention().hidden_size != decoder.hidden_size
        })
    {
        Err(Error::UnsupportedDecoderLayer(
            "model tensors or layer templates differ from decoder metadata".into(),
        ))
    } else {
        Ok(())
    }
}
