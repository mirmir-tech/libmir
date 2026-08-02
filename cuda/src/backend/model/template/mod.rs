use mircuda::bf16;
use models::layout::DecoderConfig;

use super::layer::DecoderLayerTemplate;
use crate::{
    CudaBackend, CudaTensor, DecodeMoeLayerTemplate, DenseSwiGluLayerTemplate, Error, PagedKvCache,
    Result,
    backend::{
        linear::CheckpointProjectionWeight,
        model::boundary::{ModelEmbeddingTemplate, ModelOutputHeadTemplate},
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
    logit_softcap: Option<f32>,
}

impl CudaMoeModelTemplate {
    pub(crate) fn new(
        backend: &CudaBackend,
        decoder: DecoderConfig,
        embedding: CudaTensor,
        final_norm: CudaTensor,
        output_head: CudaTensor,
        layers: Vec<DecodeMoeLayerTemplate>,
    ) -> Result<Self> {
        let layers = layers
            .into_iter()
            .map(|layer| DecoderLayerTemplate::Moe(Box::new(layer)))
            .collect();
        let scale = bf16::from_f32(f32::from(u16::try_from(decoder.hidden_size)?).sqrt()).to_f32();
        Self::new_layers(
            backend,
            decoder,
            CheckpointProjectionWeight::Dense(embedding),
            final_norm,
            CheckpointProjectionWeight::Dense(output_head),
            layers,
            scale,
        )
    }

    pub(crate) fn new_dense_bound(
        backend: &CudaBackend,
        decoder: DecoderConfig,
        embedding: CheckpointProjectionWeight,
        final_norm: CudaTensor,
        output_head: CheckpointProjectionWeight,
        layers: Vec<DenseSwiGluLayerTemplate>,
    ) -> Result<Self> {
        let layers = layers
            .into_iter()
            .map(|layer| DecoderLayerTemplate::Dense(Box::new(layer)))
            .collect();
        Self::new_layers(backend, decoder, embedding, final_norm, output_head, layers, 1.0)
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
        let logit_softcap = decoder
            .final_logit_softcapping
            .map(|value| value.to_string().parse())
            .transpose()?;
        Ok(Self {
            backend: backend.clone(),
            decoder,
            embedding,
            final_norm,
            output_head,
            layers,
            logit_softcap,
        })
    }

    #[must_use]
    pub const fn decoder(&self) -> &DecoderConfig {
        &self.decoder
    }

    pub(crate) fn allocate_shared_kv(&self) -> Result<Vec<PagedKvCache>> {
        self.layers
            .iter()
            .map(|layer| {
                let attention = layer.attention();
                self.backend.prepare_paged_kv(attention.layer, attention.cache)
            })
            .collect()
    }
}

fn validate(
    decoder: &DecoderConfig,
    final_norm: &CudaTensor,
    layers: &[DecoderLayerTemplate],
) -> Result<()> {
    if final_norm.shape() != [decoder.hidden_size]
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
