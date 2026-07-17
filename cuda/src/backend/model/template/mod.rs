use mircuda::bf16;
use models::layout::DecoderConfig;

use super::layer::DecoderLayerTemplate;
use crate::{
    CudaBackend, CudaTensor, DecodeMoeLayerTemplate, DenseSwiGluLayerTemplate, Error, PagedKvCache,
    Result, backend::output::CudaOutputHeadTemplate,
};

mod instantiate;

/// Immutable CUDA model weights and layer templates shared across sessions.
pub struct CudaMoeModelTemplate {
    backend: CudaBackend,
    decoder: DecoderConfig,
    embedding: CudaTensor,
    final_norm: CudaTensor,
    output_head: CudaOutputHeadTemplate,
    layers: Vec<DecoderLayerTemplate>,
    embedding_scale: f32,
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
        Self::new_layers(backend, decoder, embedding, final_norm, output_head, layers, scale)
    }

    pub(crate) fn new_dense(
        backend: &CudaBackend,
        decoder: DecoderConfig,
        embedding: CudaTensor,
        final_norm: CudaTensor,
        output_head: CudaTensor,
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
        embedding: CudaTensor,
        final_norm: CudaTensor,
        output_head: CudaTensor,
        layers: Vec<DecoderLayerTemplate>,
        embedding_scale: f32,
    ) -> Result<Self> {
        validate(&decoder, &embedding, &final_norm, &output_head, &layers)?;
        let output_head = CudaOutputHeadTemplate::prepare(
            backend,
            output_head,
            decoder.hidden_size,
            decoder.vocab_size,
        )?;
        Ok(Self {
            backend: backend.clone(),
            decoder,
            embedding,
            final_norm,
            output_head,
            layers,
            embedding_scale,
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
    embedding: &CudaTensor,
    final_norm: &CudaTensor,
    output_head: &CudaTensor,
    layers: &[DecoderLayerTemplate],
) -> Result<()> {
    if embedding.shape() != [decoder.vocab_size, decoder.hidden_size]
        || output_head.shape() != [decoder.vocab_size, decoder.hidden_size]
        || final_norm.shape() != [decoder.hidden_size]
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
